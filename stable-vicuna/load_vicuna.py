import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

device = torch.device("cuda:0")

tokenizer = AutoTokenizer.from_pretrained("TheBloke/stable-vicuna-13B-HF", cache_dir="./.cache/")
model = AutoModelForCausalLM.from_pretrained("TheBloke/stable-vicuna-13B-HF", cache_dir="./.cache/")
model = model.to(device=device, dtype=torch.bfloat16)
model = torch.compile(model)

