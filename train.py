import os
import numpy as np
from pathlib import Path
from datasets import load_from_disk, Dataset
from transformers import Trainer, TrainingArguments, Qwen2Tokenizer
from modeling_qwen2 import Qwen2ForCausalLM
# CUDA_VISIBLE_DEVICES=1 python train.py
# ---------- 配置 ----------
min_token = 2048
max_token = 8092
scaling_factor = 0.25
eval_num = 300
max_steps = 500

# 之前保存的 token 长度筛选缓存路径
cache_dir = Path(f"cache/16384-131072")
assert cache_dir.exists(), f"路径不存在: {cache_dir}"

# ---------- 加载模型 ----------
model_name = "Qwen/Qwen2.5-0.5B"
tokenizer = Qwen2Tokenizer.from_pretrained(model_name, trust_remote_code=True)
model = Qwen2ForCausalLM.from_pretrained(model_name, trust_remote_code=True)
model.model.rotary_emb.scaling_factor = scaling_factor

# ---------- 加载筛选后的样本 ----------
raw_dataset = load_from_disk(str(cache_dir))

# ---------- 第二阶段重新 tokenize ----------
def tokenize(example):
    tokenized = tokenizer(
        example["text"],
        truncation=True,
        padding="max_length",
        max_length=max_token,  # 避免超长输入
    )
    tokenized["labels"] = tokenized["input_ids"]
    return tokenized

# 注意保留 text 字段，以便重新 tokenize
if "text" not in raw_dataset.column_names:
    raise ValueError("请确保第一阶段保存的数据中包含 'text' 字段以供重新 tokenize")

dataset = raw_dataset.map(tokenize, remove_columns=raw_dataset.column_names)

# ---------- 统计信息 ----------
'''lengths = [len(sample["input_ids"]) for sample in dataset]
print(f"总样本数: {len(lengths)}")
print(f"平均长度: {np.mean(lengths):.2f}")
print(f"最大长度: {np.max(lengths)}")
print(f"最小长度: {np.min(lengths)}")'''

# ---------- 划分训练/验证集 ----------
split = dataset.train_test_split(test_size=0.1, seed=42)
train_dataset = split["train"]
eval_dataset = split["test"].select(range(min(eval_num, len(split["test"]))))

# ---------- 训练配置 ----------
training_args = TrainingArguments(
    output_dir=f"./step{max_steps}_scaling{scaling_factor}",
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    num_train_epochs=3,
    eval_strategy="steps",
    eval_steps=2000,
    save_strategy="steps",
    save_steps=200,
    max_steps=max_steps,
    logging_dir="./logs",
    logging_steps=20,
    save_total_limit=2,
    learning_rate=5e-5,
    bf16=True,
    report_to="none",
    remove_unused_columns=False,
    gradient_checkpointing=True,
    gradient_accumulation_steps=16,
)

trainer = Trainer(
    model=model,
    args=training_args,
    tokenizer=tokenizer,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
)

# ---------- 启动训练 ----------
trainer.train()
