
from modeling_llama import LlamaForCausalLM
from transformers import AutoTokenizer
import torch
name = "meta-llama/Llama-2-7b-chat-hf"
name = 'TinyLlama/TinyLlama-1.1B-Chat-v1.0'
tokenizer = AutoTokenizer.from_pretrained(name)
model = LlamaForCausalLM.from_pretrained(name,device_map='auto', torch_dtype=torch.bfloat16)
tokenizer.pad_token = tokenizer.bos_token
out = model(**tokenizer(['hello this is', 'hello'], return_tensors='pt', padding=True).to('cuda'), is_causal=False)
