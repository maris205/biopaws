import os
import sys
import json
import torch
import numpy as np
import evaluate
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification, 
    DataCollatorWithPadding, 
    Trainer, 
    TrainingArguments, 
    set_seed
)
from datasets import load_dataset
from tqdm import tqdm

# 设置环境变量
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

# 1. 获取命令行参数
if len(sys.argv) > 1:
    seed = int(sys.argv[1])
    try:
        lang = sys.argv[2] 
    except IndexError:
        lang = "en"
else:
    seed = 42
    lang = "en"

# 2. 设置随机种子
set_seed(seed)
result = {}
result["seed"] = seed
result["lang"] = lang

# 3. 初始化分词器和模型
model_checkpoint = "gpt2"
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForSequenceClassification.from_pretrained(model_checkpoint, num_labels=2)
model.config.pad_token_id = model.config.eos_token_id

# 定义两个专用的分词函数
def tokenize_short_function(example):
    return tokenizer(
        example["sentence1"],
        example["sentence2"],
        truncation=True,
        max_length=256,      # short 子集：完全无截断
        padding="max_length"
    )

def tokenize_full_function(example):
    return tokenizer(
        example["sentence1"],
        example["sentence2"],
        truncation=True,
        max_length=512,      # full 子集：覆盖 ~97%，最佳平衡
        padding="max_length"
    )

# 5. 加载并处理训练数据 (PAWS-X)
print(f"Loading PAWS-X ({lang}) for training...")
raw_datasets = load_dataset('paws-x', lang)
tokenized_datasets = raw_datasets.map(tokenize_short_function, batched=True)
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# 6. 定义训练参数和指标
def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return {'accuracy': (predictions == labels).sum() / len(labels)}

training_args = TrainingArguments(
    #output_dir=f"ds_job_dna_{seed}", # 动态输出目录避免冲突
    output_dir="ds_job_dna_my", # 存一个也行
    learning_rate=1e-5,
    lr_scheduler_type="constant_with_warmup",
    warmup_ratio=0.1,
    optim='adamw_torch',
    weight_decay=0.0,
    seed=seed,
    per_device_train_batch_size=64,
    per_device_eval_batch_size=64,
    num_train_epochs=4,
    eval_strategy="epoch",  # 旧版本是 evaluation_strategy，新版本必须用 eval_strategy    
    save_strategy="epoch",
    logging_strategy="epoch",
    load_best_model_at_end=True,
    save_total_limit=1 # 只保留最好的模型，节省空间
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    data_collator=data_collator,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)

# 7. 开始训练
trainer.train()

# 8. 定义通用推理函数 (复用逻辑)
# 确保模型在 GPU 上并处于评估模式
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

def run_inference(test_dataset, batch_size=64):
    preds = []
    labels = []
    
    # disable=True 禁用进度条以保持输出日志纯净
    for i in tqdm(range(0, len(test_dataset), batch_size), desc="Predicting", disable=True):
        batch = test_dataset[i : i + batch_size]
        
        inputs = {
            "input_ids": torch.tensor(batch["input_ids"]).to(device),
            "attention_mask": torch.tensor(batch["attention_mask"]).to(device),
        }
        batch_labels = batch["label"] 

        with torch.no_grad():
            outputs = model(**inputs)
            batch_preds = torch.argmax(outputs.logits, axis=-1).cpu().numpy() 

        preds.extend(batch_preds)
        labels.extend(batch_labels)
        
    metric = evaluate.load("glue", "mrpc")
    return metric.compute(predictions=preds, references=labels)

# ==========================================
# 测试集 1: protein_pair_short
# ==========================================
raw_datasets_short = load_dataset('dnagpt/biopaws', 'protein_pair_short')['train'].train_test_split(test_size=0.3, seed=seed)

# 直接分词
tokenized_raw_datasets_short = raw_datasets_short.map(tokenize_short_function, batched=True, num_proc=4)

print("Running inference on protein_pair_short set...")
ret_1 = run_inference(tokenized_raw_datasets_short["test"])
result["protein_pair_short"] = ret_1


# ==========================================
# 测试集 2: protein_pair_full (
# ==========================================
raw_datasets_full = load_dataset('dnagpt/biopaws', 'protein_pair_full')['train'].train_test_split(test_size=0.3, seed=seed)

# 直接分词 (去除了 flip_labels 以保持与基线脚本一致)
tokenized_raw_datasets_full = raw_datasets_full.map(tokenize_full_function, batched=True, num_proc=4)

print("Running inference on protein_pair_full set...")
ret_2 = run_inference(tokenized_raw_datasets_full["test"])
result["protein_pair_full"] = ret_2


# ==========================================
# 输出结果
# ==========================================
print(json.dumps(result))