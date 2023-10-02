import json
from copy import deepcopy

import torch
from tqdm import tqdm
from datasets import load_dataset
from transformers import AutoTokenizer, LlamaForCausalLM
from llama2.tokenization_llama2 import Conversation, SeparatorStyle

device = torch.device("cuda:0")

tokenizer = AutoTokenizer.from_pretrained(
    "meta-llama/Llama-2-7b-chat-hf",
    cache_dir="./.cache",
    format="pt",
    use_fast=False
)
model = LlamaForCausalLM.from_pretrained(
    "meta-llama/Llama-2-7b-chat-hf",
    cache_dir="./.cache",
    torch_dtype=torch.bfloat16
)
model = model.to(device=device)
dataset = load_dataset("glue", "sst2", split="validation")
dataset.set_format(type="pt", output_all_columns=True)

conversation = Conversation(
    name="llama-2",
    system="<s>[INST] <<SYS>>\nYou are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe. "
    "Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. "
    "Please ensure that your responses are socially unbiased and positive in nature.\n\n"
    "If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. "
    "If you don't know the answer to a question, please don't share false information.\n<</SYS>>\n\n",
    roles=("[INST]", "[/INST]"),
    messages=[],
    offset=0,
    sep_style=SeparatorStyle.LLAMA2,
    sep=" ",
    sep2=" </s><s>",
    stop_token_ids=[2],
)

ans = []
labels = []
for data in tqdm(dataset):
    conv = deepcopy(conversation)
    qs = f"For the given input text, label the sentiment of the text as positive or negative. The answer should be " \
         f"exactly 'positive' or 'negative'.\nsentence: {data['sentence']}"
    conv.append_message(conv.roles[0], qs)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()
    input_ids = tokenizer([prompt]).input_ids
    output_ids = model.generate(
        torch.as_tensor(input_ids).cuda(),
        do_sample=True,
        temperature=0.01,
        max_new_tokens=1024,
    )
    output_ids = output_ids[0][len(input_ids[0]):]
    outputs = tokenizer.decode(output_ids, skip_special_tokens=True).strip()

    ans.append(outputs)
    labels.append(data["label"].item())

with open("./.cache/llama2_sst2.json", "w") as f:
    json.dump({"ans": ans, "labels": labels}, f)
