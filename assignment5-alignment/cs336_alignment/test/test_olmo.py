import torch
from transformers import AutoTokenizer,AutoModelForCausalLM

model_name="allenai/OLMo-2-0425-1B"

tokenizer = AutoTokenizer.from_pretrained(model_name)

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",
    torch_dtype="auto"
)

prompt = "what is 2 plus 3?"

inputs = tokenizer(
    prompt,
    return_tensors="pt"
).to(model.device)

with torch.no_grad():
    outputs = model.generate(**inputs,max_new_tokens=50)

print(tokenizer.decode(
    outputs[0],
    skip_special_tokens=True
))
