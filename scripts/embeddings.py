from sentence_transformers import SentenceTransformer

# Optimal quality / speed trade-off
model = SentenceTransformer("all-MiniLM-L6-v2") 

def embed_texts(texts):
    print(f"Embedding {len(texts)} texts...")
    return model.encode(texts, convert_to_numpy=True)

# print(embed_texts(["what is the latest python release?"]))