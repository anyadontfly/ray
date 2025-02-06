import os

import torch
from transformers import LlamaForCausalLM, PreTrainedTokenizerFast


model_dir = f"{os.path.dirname(os.path.abspath(__file__))}/DeepSeek-R1-Distill-Llama-8B"
tokenizer = PreTrainedTokenizerFast(tokenizer_file=f"{model_dir}/tokenizer.json")
model = LlamaForCausalLM.from_pretrained(model_dir, torch_dtype=torch.float16)

print(model)