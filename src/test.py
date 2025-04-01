import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer, AutoModel
import psutil

tokenizer = AutoTokenizer.from_pretrained("roberta-base")
print("tokenizer loaded successfully!")

model = AutoModelForMaskedLM.from_pretrained("roberta-base")
print("Model loaded successfully!")

raw_model = AutoModel.from_pretrained("roberta-base", output_hidden_states=True, output_attentions=True)
print("raw_model loaded successfully!")

print(f"Available memory: {psutil.virtual_memory().available / (1024 * 1024 * 1024):.2f} GB")
   