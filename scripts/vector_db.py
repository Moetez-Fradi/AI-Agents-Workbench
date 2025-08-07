from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct

client = QdrantClient(host="localhost", port=6333)

COLLECTION_NAME = "ctf_docs"

def setup_collection(dim):
    client.recreate_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
    )

def upsert_embeddings(embeds):
    points = [
        PointStruct(id=i, vector=vec, payload={"text": txt})
        for i, (vec, txt) in enumerate(embeds)
    ]
    client.upsert(collection_name=COLLECTION_NAME, points=points)

def search(query_vector, top_k=3):
    print(f"Searching for top {top_k} matches...")
    return client.search(
        collection_name=COLLECTION_NAME,
        query_vector=query_vector,
        limit=top_k,
    )

# print(client.get_collections())