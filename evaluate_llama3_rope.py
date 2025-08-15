import torch
from transformers import AutoTokenizer
from modeling_llama import LlamaForCausalLM
from datasets import load_dataset, load_from_disk
import evaluate
from tqdm import tqdm
import pandas as pd
from pathlib import Path

cache_dir = Path(f"cache/16384-131072")
         
MODEL_NAME = "meta-llama/Llama-3.2-1B"
SCALING_FACTORS = [1.0, 0.85, 0.7]
CONTEXT_LENGTHS = [2048, 4096, 8192, 16384, 32768]
MAX_SAMPLES = 1500
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer.pad_token = tokenizer.eos_token
results = []
raw_dataset = load_from_disk(str(cache_dir)).select(range(MAX_SAMPLES))
                             
def get_input_ids(text, max_len):
    return tokenizer(text, return_tensors="pt", truncation=True, max_length=max_len)["input_ids"]

for sf in SCALING_FACTORS:
    print(f"\n=== Testing scaling_factor={sf} ===")
    model = LlamaForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.bfloat16, device_map = "auto")
    model.model.rotary_emb.scaling_factor = sf
    model.eval()

    for ctx_len in CONTEXT_LENGTHS:
        total_loss = 0.0
        valid_count = 0
        for sample in tqdm(raw_dataset, desc=f"Context {ctx_len}"):
            input_ids = get_input_ids(sample["text"], ctx_len).to(DEVICE)
            if input_ids.size(1) < 2:
                continue
            with torch.no_grad():
                output = model(input_ids=input_ids, labels=input_ids)
                total_loss += output.loss.item()
                valid_count += 1

        avg_loss = total_loss / max(valid_count, 1)
        ppl = torch.exp(torch.tensor(avg_loss)).item()
        print(f"[sf={sf}, len={ctx_len}] -> PPL: {ppl:.2f}")
        results.append({"scaling_factor": sf, "context_length": ctx_len, "perplexity": round(ppl, 3)})

df = pd.DataFrame(results)
df.to_csv("llama3_rope_results.csv", index=False)
print("\nResults saved to llama3_rope_results.csv")
