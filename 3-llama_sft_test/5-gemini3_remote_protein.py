import json
import random
import re
from datasets import load_dataset
from openai import OpenAI
from sklearn.metrics import accuracy_score, classification_report
from tqdm import tqdm

# ==================== API 配置 ==================== https://ai.google.dev/gemini-api/docs
API_KEY = ""
MODEL_ID = "gemini-3-flash-preview"

client = OpenAI(
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
    api_key=API_KEY,
)

# ==================== 数据准备 ====================
print("Loading dataset...")
try:
    protein_data = load_dataset('dnagpt/biopaws', 'protein_pair_remote')
    ds = protein_data['train']

    data_label_0 = [item for item in ds if item['label'] == 0]
    data_label_1 = [item for item in ds if item['label'] == 1]

    print(f"原始数据: Non-Homologous: {len(data_label_0)} | Homologous: {len(data_label_1)}")

    random.seed(42)
    sample_num = 1000 #1000个正例，1000个例
    sampled_0 = random.sample(data_label_0, min(sample_num, len(data_label_0)))
    sampled_1 = random.sample(data_label_1, min(sample_num, len(data_label_1)))

    combined_data = sampled_0 + sampled_1
    random.shuffle(combined_data)

    print(f"Data prepared: {len(combined_data)} pairs")
except Exception as e:
    print(f"Error loading dataset: {e}")
    raise

# ==================== 构建 prompt 数据 ====================
prompt_data_list = []
id_to_ground_truth = {}
for idx, item in enumerate(combined_data, 1):
    prompt_data_list.append({
        "id": idx,
        "seq_a": item['sentence1'],
        "seq_b": item['sentence2']
    })
    id_to_ground_truth[idx] = item['label']

# ==================== System Prompt ====================
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
* Do NOT provide explanations. Just the JSON array."""

# ==================== JSON 解析函数 ====================
def parse_llm_json(text):
    try:
        match = re.search(r"\[.*\]", text, re.DOTALL)
        if match:
            return json.loads(match.group(0))
        return json.loads(text)
    except Exception as e:
        print(f"JSON parse error: {e}")
        print(f"Raw response: {text[:500]}...")
        return []

# ==================== 批量推理（batch_size=10） ====================
BATCH_SIZE = 10
all_predictions = []

print(f"\n开始批量推理（每批 {BATCH_SIZE} 对）...")
for i in tqdm(range(0, len(prompt_data_list), BATCH_SIZE)):
    batch = prompt_data_list[i:i + BATCH_SIZE]

    user_prompt = f"""Here is the JSON list of {len(batch)} protein pairs to analyze.
Using your intuition about "sequence syntax" and "structural integrity," determine if each pair is "Homologous" or "Non-Homologous".
Data:
{json.dumps(batch, indent=2)}
"""

    try:
        response = client.chat.completions.create(
            model=MODEL_ID,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.1,
            #max_tokens=4096,
        )
        content = response.choices[0].message.content.strip()
        parsed = parse_llm_json(content)
        all_predictions.extend(parsed)
    except Exception as e:
        print(f"批次 {i//BATCH_SIZE + 1} 调用失败: {e}")
        # 直接退出，不继续
        print("程序退出（失败即停止，避免误导结果）")
        break  # ← 关键修改：失败就退出

print(f"推理完成，共获得 {len(all_predictions)} 个预测")

# ==================== 评估 ====================
y_true = []
y_pred = []
valid_count = 0
label_map = {"Homologous": 1, "Non-Homologous": 0}

for pred in all_predictions:
    pred_id = pred.get("id")
    pred_str = pred.get("prediction")
    if pred_id in id_to_ground_truth and pred_str in label_map:
        y_true.append(id_to_ground_truth[pred_id])
        y_pred.append(label_map[pred_str])
        valid_count += 1

if valid_count > 0:
    acc = accuracy_score(y_true, y_pred)
    print(f"\n=== 最终结果（有效预测 {valid_count} 条） ===")
    print(f"准确率: {acc:.4%}")
    print("\nClassification Report:")
    print(classification_report(
        y_true, y_pred,
        target_names=["Non-Homologous (0)", "Homologous (1)"],
        digits=4
    ))
else:
    print("无有效预测（可能所有批次都失败）")


"""
genimi3-preview推理完成，共获得 1990 个预测

=== 最终结果（有效预测 1990 条） ===
准确率: 75.5276%

Classification Report:
                    precision    recall  f1-score   support

Non-Homologous (0)     0.6999    0.8946    0.7854       996
    Homologous (1)     0.8536    0.6157    0.7154       994

          accuracy                         0.7553      1990
         macro avg     0.7767    0.7551    0.7504      1990
      weighted avg     0.7767    0.7553    0.7504      1990
"""