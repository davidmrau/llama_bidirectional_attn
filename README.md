This repo contains code to run llama with bi-directional attention by passing `is_causal=False` to the forward function.
```python3
from modeling_llama import LlamaForCausalLM
from transformers import AutoTokenizer
import torch

name = "meta-llama/Llama-2-7b-chat-hf"
tokenizer = AutoTokenizer.from_pretrained(name)
tokenizer.pad_token = tokenizer.bos_token

model = LlamaForCausalLM.from_pretrained(name, device_map='auto', torch_dtype=torch.bfloat16)

inp = tokenizer(['this is a test example', 'test'], return_tensors='pt', padding=True).to('cuda')
out = model(**inp, is_causal=False)
```
