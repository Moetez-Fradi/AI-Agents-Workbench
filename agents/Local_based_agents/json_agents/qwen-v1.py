from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch
import re
from scripts.vector_db import search
from scripts.embeddings import embed_texts
import json


def search_vector(texts):
    vector = embed_texts(texts)
    return search(vector[0])


model_path = "./Qwen3"

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
    max_memory={0: "6000MB", "cpu": "13GB"},
)


def extract_last_code_block(text):
    code_blocks = re.findall(r"<CODE>(.*?)</CODE>", text, re.DOTALL)
    if code_blocks:
        return code_blocks[-1].strip()
    return None

def run_agent(prompt):
    system = """
You are an AI assistant that answers questions by calling these tools:

1. embed_texts(texts: List[str]) → List[List[float]]
   • Takes a list of strings, returns a list of embedding‐vectors.

2. search(vectors: List[List[float]]) → str
   • Takes a list of flat float‐lists; returns a newline‐delimited
     string of the top matches.

When you want to call a tool, output *only* valid JSON, e.g.:

<JSON>
{
  "tool": "embed_texts",
  "args": { "texts": ["Who is Valbio?"] }
}
</JSON>

Once I run that, I will pass your JSON output back into you with the key `"tool_result"`.
Then you can produce your final natural‐language answer.

"""
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": prompt},
    ]
    prompt1 = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(prompt1, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out1 = model.generate(
            **inputs,
            max_new_tokens=300,
            do_sample=False,
            eos_token_id=tokenizer.eos_token_id,
        )
    full1 = tokenizer.decode(out1[0], skip_special_tokens=False)

    print("\n lena l code \n\n")
    print(full1)

    while True:
        payload = re.search(r"<JSON>(.*?)</JSON>", full1, re.DOTALL)
        if not payload:
            print("LLM didn’t output a tool call.  Answer:", full1)
            continue
        call = json.loads(payload.group(1))

        if call["tool"] == "embed_texts":
            vectors = embed_texts(**call["args"])
            tool_out = vectors
            break
        elif call["tool"] == "search":
            clean = [list(v) for v in call["args"]["vectors"]]
            tool_out = search(clean)
            break
        else:
            tool_out = f"Unknown tool {call['tool']}"
            break

    system2 = (
        """
You are a helpful assistant.
The user asked: """
        + prompt
        + """

Your search produced: """
        + tool_out
        + """

Now give a **concise**, **final answer** to the user, in plain language.
"""
    )
    messages2 = [
        {"role": "system", "content": system2},
        {"role": "user", "content": "Please summarize the result above."},
    ]
    prompt2 = tokenizer.apply_chat_template(
        messages2, tokenize=False, add_generation_prompt=True
    )
    inputs2 = tokenizer(prompt2, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out2 = model.generate(
            **inputs2,
            max_new_tokens=200,
            do_sample=False,
            eos_token_id=tokenizer.eos_token_id,
        )
    final = tokenizer.decode(out2[0], skip_special_tokens=True)
    return final

while True:
    user_input = input("enter your prompt\n")
    print(run_agent(user_input))