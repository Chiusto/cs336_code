import torch
from transformers import AutoTokenizer,AutoModelForCausalLM
from datasets import load_dataset
from pprint import pprint

model_name = "allenai/OLMo-2-0425-1B"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto",
    device_map="auto"
)
model.eval()
dataset = load_dataset("openai/gsm8k", "main")
test_data = dataset["test"]

def question_only(question):
    prompt = f"{question} Please put your final answer within \\boxed{{}}."
    return prompt

def r1_zero(question):
    prompt = f"""
    A conversation between User and Assistant. The User asks a question, and the Assistant solves it. The Assistant first thinks about the reasoning process in the mind and then provides the User with the answer. The reasoning process is enclosed within <think> </think> and the answer is enclosed within <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think> <answer> answer here </answer>.
    User: {question}
    Assistant: <think>
    """
    return prompt

def r1_zero_three_shot(question):
    prompt = f"""
    A conversation between User and Assistant. The User asks a question, and the Assistant solves it. The Assistant first thinks about the reasoning process in the mind and then provides the User with the answer. The reasoning process is enclosed within <think> </think> and answer is enclosed within <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think> <answer> answer here </answer>.
    User: There are 15 trees in the grove. Grove workers will plant trees in the grove today. After they are done, there will be 21 trees. How many trees did the grove workers plant today?
    Assistant: <think> There are 15 trees originally. Then there were 21 trees after some more were planted. So there must have been 21 - 15 = 6. So the answer is 6. </think> <answer> 6 </answer>
    User: If there are 3 cars in the parking lot and 2 more cars arrive, how many cars are in the parking lot?
    Assistant: <think> There are originally 3 cars. 2 more cars arrive. 3 + 2 = 5. So the answer is 5. </think> <answer> 5 </answer>
    User: Leah had 32 chocolates and her sister had 42. If they ate 35, how many pieces do they have left in total?
    Assistant: <think> Originally, Leah had 32 chocolates. Her sister had 42. So in total they had 32 + 42 = 74. After eating 35, they had 74 - 35 = 39. So the answer is 39. </think> <answer> 39 </answer>
    User: {question}
    Assistant: <think>
    """
    return prompt


def generate_response(prompt):
    inputs = tokenizer(
        prompt, 
        return_tensors="pt",  # 返回 PyTorch 张量
        padding=True,          # 添加 padding
        truncation=True,       # 截断过长的输入
        max_length=512        # 设置最大长度
    )
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens =512,
            do_sample=False,
        )

    input_length = inputs["input_ids"].shape[1]
    # outputs[0]里面包含：prompt token + 新生成 token，[input_length:]把前面的 prompt 去掉
    generated_ids = outputs[0][input_length:]
    response = tokenizer.decode(
        generated_ids,
        skip_special_tokens = True
    )
    return response

for i in range(10):
    print(f"\n{'=' * 80}")
    print(f"Question {i}")
    question = test_data[i]["question"]

    print("\n[Question Only]")
    print(generate_response(question_only(question)))

    print("\n[R1 Zero]")
    print(generate_response(r1_zero(question)))

    print("\n[R1 Zero 3-Shot]")
    print(generate_response(r1_zero_three_shot(question)))

#$env:UV_CACHE_DIR='C:\Users\UBU\Desktop\cs336_code\assignment5-alignment\.uv_cache_temp'
# 国内加镜像
#$env:HF_ENDPOINT='https://hf-mirror.com'
# uv run --no-sync python cs336_alignment/prompting_baselines.py
