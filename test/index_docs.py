from scripts.vector_db import search
from scripts.embeddings import embed_texts

prompt = input("give the prompt \n")
prompt_vector = embed_texts(prompt)

res = search(prompt_vector, top_k=3)

for score_point in res:
    print(score_point.payload['text'] + "\n")