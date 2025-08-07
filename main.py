from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch
import gc
from tools.googleSearch import google_search

def cut(text):
    lines = text.splitlines()

    for i, line in enumerate(lines):
        if "print" in line:
            last_paren_index = line.rfind(")")
            if last_paren_index != -1:
                lines[i] = line[:last_paren_index + 1]
                return "\n".join(lines[:i + 1])
            else:
                return "\n".join(lines[:i + 1]) 

    return text

# Mistral-7B-Instruct-v0.3

model_path = "./Qwen3"

bnb = BitsAndBytesConfig(load_in_4bit=True, llm_int4_enable_fp32_cpu_offload=True, bnb_4bit_compute_dtype=torch.float16)

torch.cuda.empty_cache()

tokenizer = AutoTokenizer.from_pretrained(model_path)

model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.float16,
    device_map="auto",
    quantization_config=bnb,
    max_memory={0: "5500MB", "cpu": "12GB"},
)

# user_prompt = "Can you be turned into an agent with smolagents ? what model are you ?"
# prompt = f"<s>[INST] {user_prompt} [/INST]"

while(True):
    prompt = input("enter your prompt \n")
    
    messages = [
        {
            "role": "system",
            "content": """
    You are a Python engineer. Write *only* pure Python code. No markdown fences, no extra quotes, no comments. Your code *have to print the result*
    exemple: user : find out if 789456 is primary
    response: def is_prime(n):
        for i in range(3, int(n**0.5) + 1, 2):
            if n % i == 0:
                return False
        return True

    print(is_prime(17))
    
    You have access to a *predefined* function:
    google_search(query: str) -> list
    This function takes a search query and returns a list of URLs from Google search.

    Use this function directly. Do not redefine it or import BeautifulSoup or requests.

    Example:
    print(google_search("latest python release"))
    """,
        },
        {"role": "user", "content": f"{prompt}"},
    ]

    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    
    inputs = tokenizer(prompt, return_tensors="pt")
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=200,
            do_sample=False,
            eos_token_id=tokenizer.eos_token_id,
        )

    result = tokenizer.decode(outputs[0], skip_special_tokens=False)
    # print(result)

    # print("Agent doing agent stuff")

    # user_prompt = "Can you be turned into an agent with smolagents ? what model are you ?"
    # prompt = f"<s>[INST] {user_prompt} [/INST]"

    try:
        content = result.split("<|im_start|>assistant")[1]
        content = cut(content)
    except IndexError:
        print("ERROR")
        
    print(content)
    exec(content, {"google_search": google_search})
    
    del inputs, outputs
    gc.collect()