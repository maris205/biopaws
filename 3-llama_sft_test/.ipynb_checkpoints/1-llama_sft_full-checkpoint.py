import os
import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    logging
)
from peft import LoraConfig
from trl import SFTTrainer, SFTConfig  # 👈 Add SFTConfig import
from huggingface_hub import login

# ================= 1. 配置区域 =================
HF_TOKEN = "hf_*****"
base_model_name = "meta-llama/Llama-3.1-8B"
new_model_name = "Llama-3.1-PAWS-En-Finetuned"

logging.set_verbosity_info()
login(token=HF_TOKEN)

# ================= 2. 准备数据集 =================
print("Loading dataset...")
dataset = load_dataset("paws-x", "en", split="train")

def format_instruction(sample):
    label_text = "Yes" if sample['label'] == 1 else "No"
    return f"""### Instruction:
Determine if the two sentences below are paraphrases of each other.

### Sentence 1:
{sample['sentence1']}

### Sentence 2:
{sample['sentence2']}

### Answer:
{label_text}"""

# ================= 3. 加载模型 =================
print("Loading model...")
model = AutoModelForCausalLM.from_pretrained(
    base_model_name,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    use_cache=False,
)

tokenizer = AutoTokenizer.from_pretrained(base_model_name)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

# ================= 4. 配置 LoRA =================
peft_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=['up_proj', 'down_proj', 'gate_proj', 'k_proj', 'q_proj', 'v_proj', 'o_proj']
)

# ================= 5. 配置训练参数 =================
# 👈 Switch to SFTConfig to support sequence length config
training_arguments = SFTConfig(
    output_dir="./results_llama3_paws",
    num_train_epochs=1,
    per_device_train_batch_size=16,
    gradient_accumulation_steps=4,
    optim="paged_adamw_32bit",
    save_steps=100,
    logging_steps=10,
    learning_rate=2e-4,
    weight_decay=0.001,
    fp16=False,
    bf16=True, 
    max_grad_norm=0.3,
    warmup_ratio=0.03,
    group_by_length=True,
    lr_scheduler_type="cosine",
    report_to="none",
    max_length=512,  # 👈 Use max_length (or try max_seq_length if this errors in your trl version)
)

# ================= 6. 开始训练 =================
print("Starting training...")

trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    peft_config=peft_config,
    processing_class=tokenizer, 
    args=training_arguments,
    formatting_func=format_instruction,
    # 👈 Remove max_seq_length from here (it's now in SFTConfig)
)

trainer.train()

# ================= 7. 保存结果 =================
print("Saving model...")
trainer.model.save_pretrained(new_model_name)
tokenizer.save_pretrained(new_model_name)
print("Done! You are ready to go.")