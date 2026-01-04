import torch
import torch.nn as nn
import numpy as np
import gc
from tqdm import tqdm
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import (
    T5Tokenizer,
    T5EncoderModel,
    Trainer,
    TrainingArguments,
    set_seed
)
from datasets import load_dataset

# ==================== 1. 基础配置 ====================
set_seed(42)
MODEL_CHECKPOINT = "Rostlab/prot_t5_xl_uniref50"
MAX_LENGTH = 256        # 针对同源判定，256通常足够且速度快
BATCH_SIZE = 16         # 96GB显存，单卡16绰绰有余
EPOCHS = 5              # 增加到5轮，给大模型充分收敛时间
LEARNING_RATE = 5e-5    # 调高学习率以激活随机初始化的分类头
OUTPUT_DIR = "./prott5_siamese_remote"

# ==================== 2. 双塔模型类 (Siamese) ====================
class ProtT5SiameseClassifier(nn.Module):
    def __init__(self, checkpoint):
        super().__init__()
        # 仅加载 Encoder 部分，节省内存并提高稳定性
        self.encoder = T5EncoderModel.from_pretrained(checkpoint, torch_dtype=torch.bfloat16)
        hidden_size = self.encoder.config.d_model # ProtT5-XL 为 1024
        
        # 定义分类头：[u; v; |u-v|; u*v] 拼接
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size * 4, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, 2)
        )

    def gradient_checkpointing_enable(self, **kwargs):
        self.encoder.gradient_checkpointing_enable(**kwargs)

    def forward(self, input_ids_1, attention_mask_1, input_ids_2, attention_mask_2, labels=None):
        # 提取两条序列的表征
        out1 = self.encoder(input_ids=input_ids_1, attention_mask=attention_mask_1).last_hidden_state
        out2 = self.encoder(input_ids=input_ids_2, attention_mask=attention_mask_2).last_hidden_state

        # Mean Pooling: 考虑 Padding
        u = self.mean_pooling(out1, attention_mask_1)
        v = self.mean_pooling(out2, attention_mask_2)

        # 特征融合：SBERT 经典拼接逻辑
        combined = torch.cat([u, v, torch.abs(u - v), u * v], dim=-1)
        
        # 转换为 float32 进行分类计算，防止 BF16 在线性层出现数值不稳定
        logits = self.classifier(combined.to(torch.float32))

        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, 2), labels.view(-1))

        return {"loss": loss, "logits": logits} if loss is not None else logits

    def mean_pooling(self, last_hidden_states, attention_mask):
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_states.size()).float()
        sum_embeddings = torch.sum(last_hidden_states * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        return sum_embeddings / sum_mask

# ==================== 3. 数据加载与并行预处理 ====================
print("正在加载数据集...")
full_dataset = load_dataset("dnagpt/biopaws", "protein_pair_remote")["train"]
sub_ds = full_dataset.train_test_split(train_size=0.2, seed=42)["train"]
ds_split = sub_ds.train_test_split(test_size=0.1, seed=42)

tokenizer = T5Tokenizer.from_pretrained(MODEL_CHECKPOINT)

def preprocess_function(examples):
    # ProtT5 必须加空格
    s1 = [" ".join(list(x)) for x in examples["sentence1"]]
    s2 = [" ".join(list(x)) for x in examples["sentence2"]]
    t1 = tokenizer(s1, padding=False, truncation=True, max_length=MAX_LENGTH)
    t2 = tokenizer(s2, padding=False, truncation=True, max_length=MAX_LENGTH)
    return {
        "input_ids_1": t1["input_ids"],
        "attention_mask_1": t1["attention_mask"],
        "input_ids_2": t2["input_ids"],
        "attention_mask_2": t2["attention_mask"],
        "labels": examples["label"]
    }

tokenized_ds = ds_split.map(preprocess_function, batched=True, remove_columns=ds_split["train"].column_names, num_proc=4)

# 自定义 Data Collator 处理双序列填充
def siamese_collate_fn(features):
    batch = {}
    for i in [1, 2]:
        ids = [torch.tensor(f[f"input_ids_{i}"]) for f in features]
        masks = [torch.tensor(f[f"attention_mask_{i}"]) for f in features]
        batch[f"input_ids_{i}"] = torch.nn.utils.rnn.pad_sequence(ids, batch_first=True, padding_value=tokenizer.pad_token_id)
        batch[f"attention_mask_{i}"] = torch.nn.utils.rnn.pad_sequence(masks, batch_first=True, padding_value=0)
    batch["labels"] = torch.tensor([f["labels"] for f in features])
    return batch

# ==================== 4. 训练配置 ====================
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)
    return {"accuracy": accuracy_score(labels, preds)}

model = ProtT5SiameseClassifier(MODEL_CHECKPOINT)

training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    learning_rate=LEARNING_RATE,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    num_train_epochs=EPOCHS,
    weight_decay=0.01,
    lr_scheduler_type="cosine",       # 余弦退火有助于大模型收敛
    warmup_ratio=0.1,                 # 前10%步数预热
    bf16=True,
    gradient_checkpointing=True,
    gradient_accumulation_steps=2,    # 等效 BatchSize = 32
    eval_strategy="epoch",
    save_strategy="epoch",
    save_total_limit=3,
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    remove_unused_columns=False,      # 必须设为 False
    label_names=["labels"],
    report_to="none"
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_ds["train"],
    eval_dataset=tokenized_ds["test"],
    data_collator=siamese_collate_fn,
    compute_metrics=compute_metrics,
)

# ==================== 5. 执行 ====================
print("\n开始双塔微调...")
trainer.train()

print("\n=== 独立测试集评估 ===")
predictions = trainer.predict(tokenized_ds["test"])
y_pred = np.argmax(predictions.predictions, axis=1)
y_true = predictions.label_ids
print(classification_report(y_true, y_pred, target_names=["Non-Homologous", "Homologous"]))


"""
Classification Report:
                precision    recall  f1-score   support

Non-Homologous       0.76      0.04      0.08      3590
    Homologous       0.51      0.99      0.67      3598

      accuracy                           0.51      7188
     macro avg       0.64      0.51      0.37      7188
  weighted avg       0.64      0.51      0.37      7188
"""