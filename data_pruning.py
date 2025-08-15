import os
import json
from pathlib import Path
from datasets import Dataset, concatenate_datasets
from transformers import Qwen2Tokenizer

# 参数配置
min_token = 2048
max_token = 32768
model_name = "Qwen/Qwen2.5-0.5B"
save_dir = f"cache/{min_token}-{max_token}"

# 初始化 tokenizer
tokenizer = Qwen2Tokenizer.from_pretrained(model_name, trust_remote_code=True)

# 要处理的数据文件
base_dir = Path(__file__).parent
files = [
    # "arxiv_sample.jsonl",           
    # "book_sample.jsonl",            
    # "stackexchange_sample.jsonl",   
    # "wikipedia_sample.jsonl",       
    "cc_2023-06_sample.jsonl",       
    "cc_2022-05_sample.jsonl",
    "cc_2021-04_sample.jsonl",
]

datasets = []

# tokenize 只是为了统计 token 长度
def tokenize_and_count(example):
    tokens = tokenizer(example["text"], truncation=False, padding=False)
    example["token_len"] = len(tokens["input_ids"])
    return example

# 读取 + 筛选流程
for fname in files:
    path = base_dir / fname
    with open(path, "r", encoding="utf-8") as f:
        data = [{"text": json.loads(line)["text"]} for line in f if "text" in json.loads(line)]

    ds = Dataset.from_list(data)

    # Tokenize 仅用于统计长度
    ds = ds.map(tokenize_and_count)

    # 过滤掉长度不合适的样本
    ds = ds.filter(lambda x: min_token < x["token_len"] < max_token)

    # 保留 text 字段（其余字段如 token_len 都删掉）
    ds = ds.remove_columns([col for col in ds.column_names if col != "text"])

    print(f"{fname} 保留样本数: {len(ds)}")
    datasets.append(ds)

# 合并全部数据
combined = concatenate_datasets(datasets)

# 保存为 HuggingFace Dataset（下一阶段可 load_from_disk 使用）
combined.save_to_disk(save_dir)
print(f"所有合并后样本数: {len(combined)}")
