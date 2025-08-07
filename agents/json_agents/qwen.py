from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch
import re
from scripts.vector_db import search
from scripts.embeddings import embed_texts
import json
import gc
# from tools.googleSearch import google_search


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


SYSTEM_PROMPT = """
You are an AI assistant that answers questions about Valbio, a Biodegradable plastic startup, by calling this tool:

1. search_vector(texts: List[str]) → str  
   • Takes a list of search-ready strings; returns top matches from a knowledge base.

Your job is to:
- Understand the user's question
- Improve, rephrase, or clarify it for more effective retrieval
- Then call the tool with a valid JSON block

IMPORTANT:
- You MUST return only one JSON object between <JSON> ... </JSON> tags.
- The "texts" field must contain a refined version of the user's question — not the original unless it is already optimal.
- The JSON must be parsable and not contain commentary.

### Examples

User: “What are Valbio's added values?”  
Assistant:
<JSON>
{
  "tool": "search_vector",
  "args": {
    "texts": ["What are the unique contributions and value propositions of Valbio?"]
  }
}
</JSON>

---

User: “Why buy injectable plastic from Valbio?”  
Assistant:
<JSON>
{
  "tool": "search_vector",
  "args": {
    "texts": ["What are the main advantages of choosing Valbio's injectable plastic products?"]
  }
}
</JSON>

---

User: “future plans?”  
Assistant:
<JSON>
{
  "tool": "search_vector",
  "args": {
    "texts": ["What are Valbio's future plans and strategic goals?"]
  }
}
</JSON>

---

Now, whenever a user sends a question, think carefully, rephrase it if needed, and call the tool using only a valid JSON block as shown.


"""

SYSTEM_PROMPT2 = """
You are an AI assistant that has just retrieved information via a tool and must now craft a clear, concise, human-friendly response in **the exact language used by the user**.

You will receive two sections:

<USER>
…The user’s original question…
</USER>

<TOOL_RESULT>
…A newline-delimited list of raw search hits from the knowledge base…
</TOOL_RESULT>

Your task:
1. Read the user’s question and keep its intent in mind.
2. Analyze the TOOL_RESULT entries and select the most relevant facts.
3. Compose a single, coherent answer in the same language as the user’s question—do not switch languages.
4. Do NOT output any JSON, system tags, or mention the tool—you’re simply answering.

Guidelines:
- Begin with a direct answer to the user’s question.
- If TOOL_RESULT is empty or insufficient, politely ask a follow-up question for clarification.
- Keep the tone professional and concise (1–3 short paragraphs).

### Examples

<USER>
¿Qué hace a Valbio especial?
</USER>
<TOOL_RESULT>
La plataforma de Valbio combina cribado de alto rendimiento con análisis impulsados por IA.
Ofrece seguimiento completo de muestras, control de calidad en tiempo real y protocolos personalizables.
</TOOL_RESULT>

—→ Respuesta final:
Valbio destaca por unir el cribado a gran escala con análisis basados en IA para acelerar la I+D. Además, ofrece un seguimiento completo de muestras para garantizar la integridad de los datos y control de calidad en tiempo real para detectar errores al instante.

---

<USER>
What are Valbio’s future plans?
</USER>
<TOOL_RESULT>
Valbio plans to expand its biodegradable polymer line into medical packaging.
They’re partnering with two major hospitals for clinical trials in early 2026.
</TOOL_RESULT>

—→ Final answer:
Valbio aims to broaden its biodegradable polymer range specifically for medical packaging, and it has secured partnerships with two leading hospitals to begin clinical trials in early 2026.
"""

def llm_generate(prompt: str, max_new_tokens: int = 200) -> str:
    try:
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                use_cache=False,  # or False if short gen
            )
    except torch.cuda.OutOfMemoryError:
        print("⚠️ Out of memory on GPU. Falling back to CPU.")
        torch.cuda.empty_cache()

        # Move model to CPU
        model.to("cpu")

        # Move inputs to CPU
        inputs = tokenizer(prompt, return_tensors="pt").to("cpu")
        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                use_cache=False,  # to save RAM on CPU
            )

    return tokenizer.decode(out[0], skip_special_tokens=False)


def run_agent(user_question: str) -> str:
    prompt1 = tokenizer.apply_chat_template(
        [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_question},
        ],
        tokenize=False,
        add_generation_prompt=True,
    )
    print("generating first response...")
    raw1 = llm_generate(prompt1, max_new_tokens=100)
    del prompt1
    torch.cuda.empty_cache()
    gc.collect()
    
    print("raw1:", raw1)

    matches = re.findall(r"<JSON>\s*(\{.*?\})\s*</JSON>", raw1, re.DOTALL)
    if not matches:
        return f"🛑 Model did not output a tool call JSON.\n\n{raw1}"

    json_block = matches[-1].strip()

    print("RAW JSON block:")
    print(json_block)

    try:
        call = json.loads(json_block)
    except json.JSONDecodeError as e:
        return f"❌ Failed to parse JSON: {e}\n\nRaw block:\n{json_block}"


    print("tool call:", call)

    if call["tool"] == "search_vector":
        tool_output = search_vector(call["args"]["texts"])

    else:
        tool_output = f"Unknown tool: {call['tool']}"

    print("tool_output:", tool_output)

    tool_result = ""
    for score_point in tool_output:
        tool_result += score_point.payload["text"] + "\n"
    
    print("tool_output:", tool_output)

    followup_prompt = (
        SYSTEM_PROMPT2
        + "\n<USER>\n"
        + user_question
        + "\n"
        + "<TOOL_RESULT>\n"
        + json.dumps(tool_result)
        + "\n</TOOL_RESULT>\n"
    )

    torch.cuda.empty_cache()
    print("generating follow-up response...")
    raw2 = llm_generate(
        tokenizer.apply_chat_template(
            [
                {"role": "system", "content": followup_prompt},
                {"role": "user", "content": "Please give me a concise, final answer."},
            ],
            tokenize=False,
            add_generation_prompt=True,
        ),
        max_new_tokens=128,
    )
    return re.sub(r"<.*?>", "", raw2).strip()


if __name__ == "__main__":
    print("🔎 Qdrant-Agent ready. Type your question and press Enter.")
    try:
        while True:
            q = input("\nYou> ").strip()
            if not q:
                continue
            answer = run_agent(q)
            print("\nAgent>", answer)
    except (KeyboardInterrupt, EOFError):
        print("\nGoodbye!")


# Version 1

# def extract_last_code_block(text):
#     code_blocks = re.findall(r"<CODE>(.*?)</CODE>", text, re.DOTALL)
#     if code_blocks:
#         return code_blocks[-1].strip()
#     return None

# def run_agent(prompt):
#     system = """
# You are an AI assistant that answers questions by calling these tools:

# 1. embed_texts(texts: List[str]) → List[List[float]]
#    • Takes a list of strings, returns a list of embedding‐vectors.

# 2. search(vectors: List[List[float]]) → str
#    • Takes a list of flat float‐lists; returns a newline‐delimited
#      string of the top matches.

# When you want to call a tool, output *only* valid JSON, e.g.:

# <JSON>
# {
#   "tool": "embed_texts",
#   "args": { "texts": ["Who is Valbio?"] }
# }
# </JSON>

# Once I run that, I will pass your JSON output back into you with the key `"tool_result"`.
# Then you can produce your final natural‐language answer.

# """
#     messages = [
#         {"role": "system", "content": system},
#         {"role": "user", "content": prompt},
#     ]
#     prompt1 = tokenizer.apply_chat_template(
#         messages, tokenize=False, add_generation_prompt=True
#     )
#     inputs = tokenizer(prompt1, return_tensors="pt").to(model.device)
#     with torch.no_grad():
#         out1 = model.generate(
#             **inputs,
#             max_new_tokens=300,
#             do_sample=False,
#             eos_token_id=tokenizer.eos_token_id,
#         )
#     full1 = tokenizer.decode(out1[0], skip_special_tokens=False)

#     print("\n lena l code \n\n")
#     print(full1)

#     while True:
#         payload = re.search(r"<JSON>(.*?)</JSON>", full1, re.DOTALL)
#         if not payload:
#             print("LLM didn’t output a tool call.  Answer:", full1)
#             continue
#         call = json.loads(payload.group(1))

#         # 3) exec the tool
#         if call["tool"] == "embed_texts":
#             vectors = embed_texts(**call["args"])
#             tool_out = vectors
#             break
#         elif call["tool"] == "search":
#             # ensure pure python list, not np.array
#             clean = [list(v) for v in call["args"]["vectors"]]
#             tool_out = search(clean)
#             break
#         else:
#             tool_out = f"Unknown tool {call['tool']}"
#             break

#     # code = extract_last_code_block(full1)
#     # safe_code = code.encode("ascii", "ignore").decode("ascii")

#     # print("\n lena l code \n\n")
#     # print(safe_code)

#     # old_stdout = sys.stdout
#     # sys.stdout = StringIO()
#     # try:
#     #     exec(safe_code, {"embed_texts": embed_texts, "search": search})
#     #     result = sys.stdout.getvalue().strip()
#     # except Exception as e:
#     #     result = f"ERROR: {e}"
#     # finally:
#     #     sys.stdout = old_stdout

#     system2 = (
#         """
# You are a helpful assistant.
# The user asked: """
#         + prompt
#         + """

# Your search produced: """
#         + tool_out
#         + """

# Now give a **concise**, **final answer** to the user, in plain language.
# """
#     )
#     messages2 = [
#         {"role": "system", "content": system2},
#         {"role": "user", "content": "Please summarize the result above."},
#     ]
#     prompt2 = tokenizer.apply_chat_template(
#         messages2, tokenize=False, add_generation_prompt=True
#     )
#     inputs2 = tokenizer(prompt2, return_tensors="pt").to(model.device)
#     with torch.no_grad():
#         out2 = model.generate(
#             **inputs2,
#             max_new_tokens=200,
#             do_sample=False,
#             eos_token_id=tokenizer.eos_token_id,
#         )
#     final = tokenizer.decode(out2[0], skip_special_tokens=True)
#     return final


# — in your REPL loop —
# while True:
#     user_input = input("enter your prompt\n")
#     print(run_agent(user_input))
