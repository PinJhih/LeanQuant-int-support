import torch
from leanquant import LeanQuantModelForCausalLM
from transformers import AutoTokenizer

### Load model and tokenizer
base_model_name = "meta-llama/Llama-3.1-8B-Instruct"
model = LeanQuantModelForCausalLM.from_pretrained(
    base_model_name,
    "./llama3.1.8b.4bit.safetensors",
    bits=4,
    device_map="auto"
)
model.eval()
tokenizer = AutoTokenizer.from_pretrained(base_model_name)

### Tokenize prompt
prompt = [
    {"role": "system", "content": "You are a helpful assistant, that responds as a pirate."},
    {"role": "user", "content": "What is quantization for deep learning models?"},
]
inputs = tokenizer.apply_chat_template(
    prompt,
    tokenize=True,
    add_generation_prompt=True,
    return_tensors="pt",
    return_dict=True,
).to(model.device)

### Run generation and decode generated tokens
with torch.no_grad():
    output = model.generate(**inputs, do_sample=True, max_new_tokens=256)

generated_text = tokenizer.decode(output[0], skip_special_tokens=False)
print(generated_text)