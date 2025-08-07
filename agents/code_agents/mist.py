from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch
import gc
from tools.search import free_search


def cut(text):
    lines = text.splitlines()

    for i, line in enumerate(lines):
        if "print" in line:
            last_paren_index = line.rfind(")")
            if last_paren_index != -1:
                lines[i] = line[: last_paren_index + 1]
                return "\n".join(lines[: i + 1])
            else:
                return "\n".join(lines[: i + 1])

    return text


model_path = "./Mistral7B"

bnb = BitsAndBytesConfig(
    load_in_4bit=True,
    llm_int4_enable_fp32_cpu_offload=True,
    bnb_4bit_compute_dtype=torch.float16,
)

torch.cuda.empty_cache()

tokenizer = AutoTokenizer.from_pretrained(model_path)

model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.float16,
    device_map="auto",
    quantization_config=bnb,
    max_memory={0: "5500MB", "cpu": "12GB"},
)

while True:
    prompt = input("enter your prompt \n")

    messages = [
        {
            "role": "system",
            "content": """
You are a Python engineer.  
Your job: Output *only* valid, executable Python code — no markdown, no quotes, no explanations.

Rules:
1. Your code must run without modification.
2. **Always include all necessary imports** explicitly at the top of your code. Never assume a module is already imported.
3. Always use `print(...)` to show the final result.
4. You can use the predefined function: `free_search(query: str) -> list`  
   - It returns a list of clean URLs from DuckDuckGo search.  
   - Do not redefine it.
5. Only use standard libraries plus `requests` and `bs4` (BeautifulSoup) when scraping HTML.

Workflow:
1. If a search is needed, call `free_search()` and store the result.
2. Import every external library you use (e.g., `import requests`, `from bs4 import BeautifulSoup`).
3. Process the result and `print` the output.

Good Example:
urls = free_search("latest python release")
import requests
from bs4 import BeautifulSoup
html = requests.get(urls[0]).text
soup = BeautifulSoup(html, "html.parser")
print(soup.title.text)
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
            max_new_tokens=100,
            do_sample=False,
            eos_token_id=tokenizer.eos_token_id,
        )

    result = tokenizer.decode(outputs[0], skip_special_tokens=False)

    try:
        content = (result.split("[/INST] ```python")[1]).split("```")[0]
        content = cut(content)
        print(content)
    except IndexError:
        print("ERROR")
    except Exception as e:
        print(f"An error occurred: {e}")

    print(content)
    exec(content, {"free_search": free_search})

    del inputs, outputs
    gc.collect()
