from scripts.vector_db import setup_collection, upsert_embeddings
from scripts.embeddings import embed_texts
from scripts.read_data import load_cheats_from_folder

# This function should be modified depending on the data, like changing the length of 
# the text or the separation method
cheats = load_cheats_from_folder(folder="data/cheat")

vectors = embed_texts(cheats)
setup_collection(dim=vectors.shape[1])
upsert_embeddings(zip(vectors, cheats))