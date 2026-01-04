#!/usr/bin/env python
# coding: utf-8

# In[1]:


# import os

# # 设置环境变量
# os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

# # 打印环境变量以确认设置成功
# print(os.environ.get('HF_ENDPOINT'))

# import subprocess
# import os

# result = subprocess.run('bash -c "source /etc/network_turbo && env | grep proxy"', shell=True, capture_output=True, text=True)
# output = result.stdout
# for line in output.splitlines():
#     if '=' in line:
#         var, value = line.split('=', 1)
#         os.environ[var] = value


# In[2]:


import os
import subprocess
import json
import random
import re
import torch  # 新增
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig # 新增
from huggingface_hub import login # 新增
from sklearn.metrics import accuracy_score, classification_report


# In[3]:


# ==========================================
# 3. 加载模型 (Base Model + LoRA Adapter)
# ==========================================
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel # 👈 必须引入这个库
from huggingface_hub import login

# 定义路径
BASE_MODEL_ID = "meta-llama/Llama-3.1-8B"       # 原始底座
ADAPTER_PATH = "./Llama-3.1-8B-PAWS-En-Finetuned" # 你刚训练好的微调结果
HF_TOKEN = ""

print(f"Logging in...")
login(token=HF_TOKEN)


# In[ ]:


print(f"Loading Base Model: {BASE_MODEL_ID}...")
# 1. 先加载底座模型
base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL_ID,
    torch_dtype=torch.float16, # 4090 上推理推荐用半精度
    device_map="auto",
)

print(f"Loading LoRA Adapter: {ADAPTER_PATH}...")
# 2. 关键步骤：把微调的补丁“挂”到底座上
model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)

print("Loading Tokenizer...")
# 3. 加载分词器 (通常直接用底座的，或者你保存目录里的)
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID)

# 设定 Llama 3 的终止符
terminators = [
    tokenizer.eos_token_id,
    tokenizer.convert_tokens_to_ids("<|eot_id|>")
]

print("Model loaded successfully! Ready for inference.")


# In[9]:


# ==========================================
# 2. 数据准备 (加载、采样、打乱)
# ==========================================
print("Loading dataset...")
try:
    # 尝试加载数据集
    protein_data = load_dataset('dnagpt/biopaws', 'protein_pair_short')
    # local_dataset_path = "./biopaws" 
    # # 修改这里：第一个参数改为本地路径
    # protein_data = load_dataset(
    #     local_dataset_path,          # 👈 这里改成你的本地文件夹路径
    #     'protein_sim_pair_450bp',    # 配置名保持不变
    #     trust_remote_code=True       # 👈 加上这个，允许执行本地文件夹里的加载脚本
    # )

    # 方法：直接从 CSV 文件加载（推荐，最简单）
    # 假设你的 CSV 文件有三列：sentence1, sentence2, label
    #csv_path = "protein_pair_sample_200.csv" #protein_pair_sample_200_length_restricted.csv  protein_pair_sample_200.csv 
    
    # 使用 load_dataset 直接读取本地 CSV
    #protein_data = load_dataset("csv", data_files=csv_path)
    
    ds = protein_data['train']
    
    # 分离数据
    data_label_0 = [item for item in ds if item['label'] == 0]
    data_label_1 = [item for item in ds if item['label'] == 1]
    
    # 随机采样 (各30%)
    random.seed(42)
    sample_num_0 = int(len(data_label_0) * 0.3)
    sample_num_1 = int(len(data_label_1) * 0.3)
    sampled_0 = random.sample(data_label_0, sample_num_0)
    sampled_1 = random.sample(data_label_1, sample_num_1)
    
    # 合并并打乱
    combined_data = sampled_0 + sampled_1
    random.shuffle(combined_data)
    
    print(f"Data prepared: {len(combined_data)} pairs.")

except Exception as e:
    print(f"Error loading dataset: {e}")
    
# 构建用于 Prompt 的 JSON List，并建立 ID -> Label 的映射用于后续验证
prompt_data_list = []
id_to_ground_truth = {}

for idx, item in enumerate(combined_data, 1):
    prompt_data_list.append({
        "id": idx,
        "seq_a": item['sentence1'],
        "seq_b": item['sentence2']
    })
    id_to_ground_truth[idx] = item['label']




# ==========================================
# 3. 构建 Prompt (System vs User)
# ==========================================

# System Prompt: 定义规则、类比和输出格式
system_prompt = """You are an expert bioinformatics assistant capable of linguistic transfer learning.

1. The Concept:
In English, a sentence can be rearranged structurally but keep the same meaning (Paraphrase). Or, it can be scrambled to lose its logic (Adversarial).

2. The Analogy:
* **Homologous Proteins** are like **Paraphrases**: They have different sequences due to evolution, but they fold into the same structure and function.
* **Non-Homologous/Random Proteins** are like **Adversarial Sentences**: They look like proteins, but their internal structural logic is broken.

3. Output Requirements:
* I will provide a JSON list of protein pairs.
* You must return a RAW JSON object containing a list of results.
* The format must be strictly: `[{"id": 1, "prediction": "Homologous"}, {"id": 2, "prediction": "Non-Homologous"}, ...]`
* Do NOT provide explanations. Just the JSON array.
"""


# In[12]:


# ==========================================
# 5. 解析结果与评估 (移动到这里，以便在循环中使用)
# ==========================================
def parse_llm_json(text):
    """提取并解析 JSON"""
    try:
        # 寻找 JSON 数组 [...]
        match = re.search(r"\[.*\]", text, re.DOTALL)
        if match:
            return json.loads(match.group(0))
        return json.loads(text)
    except Exception:
        return []


# ==========================================
# 4. 执行本地推理 (针对 Base 模型的修改版)
# ==========================================
print("-" * 30)
print(f"Running inference ...")

# 初始化总体预测列表
predictions_list = []

# 批次大小：最多20个序列对（10+10）
batch_size = 10 #这个太小效果也不好，为啥呢？

# 计算批次数
num_batches = (len(prompt_data_list) + batch_size - 1) // batch_size

for batch_idx in range(num_batches):
    start = batch_idx * batch_size
    end = start + batch_size
    prompt_data_batch = prompt_data_list[start:end]
    
    if not prompt_data_batch:
        continue
    
    print(f"Processing batch {batch_idx + 1}/{num_batches} with {len(prompt_data_batch)} pairs...")
    
    # User Prompt: 提供具体数据（动态数量）
    user_prompt = f"""Here is the JSON list of {len(prompt_data_batch)} protein pairs to analyze.
Using your intuition about "sequence syntax" and "structural integrity," determine if each pair is "Homologous" or "Non-Homologous".

Data:
{json.dumps(prompt_data_batch, indent=2)}
"""
    
    # --- 关键修改 1: 手动拼接 Prompt，不用 Chat Template ---
    # Base 模型需要你像写文章开头一样引导它
    # 我们在最后强行加一个 ```json\n[，诱导它直接开始写 JSON 数组
    raw_prompt = f"""
{system_prompt}

{user_prompt}

The results in JSON format are:
```json
[
"""

    # --- 关键修改 2: 直接编码字符串 ---
    input_ids = tokenizer(
        raw_prompt, 
        return_tensors="pt"
    ).input_ids.to(model.device)

    print("Generating response for batch...")

    try:
        with torch.no_grad():
            outputs = model.generate(
                input_ids,
                max_new_tokens=4096,
                # --- 关键修改 3: Base 模型只要遇到 EOS 就停止 ---
                eos_token_id=tokenizer.eos_token_id, 
                pad_token_id=tokenizer.eos_token_id,
                do_sample=True,
                temperature=0.1,
                top_p=0.9,
            )

        # 解码
        response = outputs[0][input_ids.shape[-1]:]
        full_content = tokenizer.decode(response, skip_special_tokens=True)
        
        # --- 关键修改 4: 因为我们在 Prompt 里手动加了开头，这里要补回来 ---
        # 这样后续的 JSON 解析器才能读懂
        full_content = "[\n" + full_content 

        print("Response received for batch.")
        print(f"Response snippet: {full_content[:200]}...")

        # 解析批次预测
        predictions_batch = parse_llm_json(full_content)
        predictions_list.extend(predictions_batch)

    except Exception as e:
        print(f"Inference Failed for batch {batch_idx + 1}: {e}")



# 映射与计算
label_map = {"Homologous": 1, "Non-Homologous": 0}
y_true = []
y_pred = []

print("-" * 30)
if not predictions_list:
    print("Failed to parse JSON from model response.")
else:
    print(f"Parsed {len(predictions_list)} predictions.")
    
    for item in predictions_list:
        p_id = item.get('id')
        p_str = item.get('prediction') 
        
        # 确保 ID 存在且预测值有效
        if p_id in id_to_ground_truth and p_str in label_map:
            y_true.append(id_to_ground_truth[p_id])
            y_pred.append(label_map[p_str])

    # 输出最终指标
    if y_true:
        acc = accuracy_score(y_true, y_pred)
        print(f"\nFinal Accuracy: {acc:.2%}")
        print("\nClassification Report:")
        print(classification_report(y_true, y_pred, target_names=["Non-Homologous (0)", "Homologous (1)"]))
        
        # 保存结果用于后续分析
        result_log = {
            "model": "llama3.1",
            "accuracy": acc,
            "predictions": predictions_list
        }
        print(acc)
        # with open("doubao_result.json", "w") as f:
        #     json.dump(result_log, f, indent=2)
        #     print("Results saved to 'doubao_result.json'")
    else:
        print("No valid matching IDs found between Prompt and Response.")


# In[ ]:


"""
------------------------------
Parsed 900 predictions.

Final Accuracy: 72.00%

Classification Report:
                    precision    recall  f1-score   support

Non-Homologous (0)       0.67      0.86      0.75       450
    Homologous (1)       0.81      0.58      0.67       450

          accuracy                           0.72       900
         macro avg       0.74      0.72      0.71       900
      weighted avg       0.74      0.72      0.71       900

0.72

"""