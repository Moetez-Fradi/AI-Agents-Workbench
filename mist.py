from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch
from tools.googleSearch import google_search
import sys
from io import StringIO
import re

def extract_last_code_block(text):
    code_blocks = re.findall(r"<CODE>(.*?)</CODE>", text, re.DOTALL)
    if code_blocks:
        return code_blocks[-1].strip() 
    return None

model_path = "./Mistral7B"

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

def run_agent(prompt):
    # 1) FIRST PASS: “think” + code
    system = """
You are a Python engineer. Write only pure Python code (ASCII only).  
Whenever you fetch HTML via urllib.request.urlopen().read(), you must decode the bytes to a str with `.decode("utf-8")` before using regex.  
- First, think (briefly) about what to do.
- Then write pure Python code that prints the result.
- Do NOT execute the code yourself—just output it under a <CODE> block.
Example format:

<THOUGHT>
I need to compute X because…
</THOUGHT>
<CODE>
def foo(): …
print(foo())
</CODE>
"""
    messages = [
        {"role":"system", "content": system},
        {"role":"user",   "content": prompt},
    ]
    prompt1 = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt1, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out1 = model.generate(**inputs, max_new_tokens=300, do_sample=False,
                              eos_token_id=tokenizer.eos_token_id)
    full1 = tokenizer.decode(out1[0], skip_special_tokens=False)
    
    print("\n lena l code \n\n")
    print(full1)

    code = extract_last_code_block(full1)
    safe_code = code.encode("ascii", "ignore").decode("ascii")
    
    print("\n lena l code \n\n")
    print(safe_code)
    
    # capture stdout
    old_stdout = sys.stdout
    sys.stdout = StringIO()
    try:
        exec(safe_code, {"google_search": google_search})
        result = sys.stdout.getvalue().strip()
    except Exception as e:
        result = f"ERROR: {e}"
    finally:
        sys.stdout = old_stdout

    system2 = """
You are a helpful assistant. 
The user asked: """ + prompt + """

Your code produced: """ + result + """

Now give a **concise**, **final answer** to the user, in plain language. Do NOT re-output code.
"""
    messages2 = [
        {"role":"system", "content": system2},
        {"role":"user",   "content": "Please summarize the result above."},
    ]
    prompt2 = tokenizer.apply_chat_template(messages2, tokenize=False, add_generation_prompt=True)
    inputs2 = tokenizer(prompt2, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out2 = model.generate(**inputs2, max_new_tokens=200, do_sample=False,
                              eos_token_id=tokenizer.eos_token_id)
    final = tokenizer.decode(out2[0], skip_special_tokens=True)
    return final

# — in your REPL loop —
while True:
    user_input = input("enter your prompt\n")
    print(run_agent(user_input))
