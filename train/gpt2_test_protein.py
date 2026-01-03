import os
import sys
import json
import torch
import numpy as np
import evaluate
from transformers import AutoTokenizer, AutoModelForSequenceClassification, set_seed
from datasets import load_dataset
from tqdm import tqdm

# 设置环境变量
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

# 获取命令行参数
if len(sys.argv) > 1:
    seed = int(sys.argv[1])
    try:
        lang = sys.argv[2] 
    except IndexError:
        lang = "en"
else:
    seed = 42
    lang = "en"

# 设置随机种子
set_seed(seed)

result = {}
result["seed"] = seed
result["type"] = "no_finetune_baseline"

# 初始化模型和分词器
model_checkpoint = "gpt2"
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
tokenizer.pad_token = tokenizer.eos_token

# 加载模型 (预训练权重 + 随机分类头)
model = AutoModelForSequenceClassification.from_pretrained(model_checkpoint, num_labels=2)
model.config.pad_token_id = model.config.eos_token_id

# 移动到 GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

# 定义分词函数
def tokenize_function(example):
    return tokenizer(
        example["sentence1"], 
        example["sentence2"], 
        truncation=True,
        max_length=256, 
        padding="max_length"
    )

# 定义推理函数
def run_inference(test_dataset, batch_size=64):
    preds = []
    labels = []
    
    # disable=True 禁用进度条以保持输出纯净
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
# 测试集 1: protein_sim_pair_450bp
# ==========================================
raw_datasets_1 = load_dataset('dnagpt/gene_lan_transfer', 'protein_sim_pair_450bp')['train'].train_test_split(test_size=0.3, seed=seed)

tokenized_datasets_1 = raw_datasets_1.map(tokenize_function, batched=True, num_proc=4)

ret_1 = run_inference(tokenized_datasets_1["test"])
result["protein_pair_150"] = ret_1

# ==========================================
# 测试集 2: protein_sim_pair_150bp
# ==========================================
raw_datasets_2 = load_dataset('dnagpt/gene_lan_transfer', 'protein_sim_pair_150bp')['train'].train_test_split(test_size=0.3, seed=seed)

# 直接分词，不再翻转标签
tokenized_datasets_2 = raw_datasets_2.map(tokenize_function, batched=True, num_proc=4)

ret_2 = run_inference(tokenized_datasets_2["test"])
result["protein_pair_50"] = ret_2

# ==========================================
# 输出结果
# ==========================================
print(json.dumps(result))