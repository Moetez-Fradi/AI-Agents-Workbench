# Retrieval-Augmented Generation with LlamaIndex and ChromaDB
# Default RAG query engine

from llama_index.llms.openai_like import OpenAILike
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.ingestion import IngestionPipeline
from llama_index.core import VectorStoreIndex
from dotenv import load_dotenv
import os
from llama_index.core import Document
import asyncio
import chromadb
from llama_index.vector_stores.chroma import ChromaVectorStore
import warnings
from llama_index.core.evaluation import FaithfulnessEvaluator

warnings.filterwarnings("ignore")
load_dotenv()

db = chromadb.PersistentClient(path="./db_chroma")
chroma_collection = db.get_or_create_collection("alfred")
vector_store = ChromaVectorStore(chroma_collection=chroma_collection)

embedder = HuggingFaceEmbedding(model_name="all-MiniLM-L6-v2")
index = VectorStoreIndex.from_vector_store(vector_store, embed_model=embedder)

pipeline = IngestionPipeline(
    transformations=[
        SentenceSplitter(chunk_overlap=0),
        HuggingFaceEmbedding(model_name="all-MiniLM-L6-v2"),
    ],
    vector_store=vector_store,
)

async def main():
    result = await pipeline.arun(documents=[Document.example()])
    return result
asyncio.run(main())

llm = OpenAILike(
    model="meta-llama/llama-3-8b-instruct",
    api_base=os.getenv("OPENAI_BASE_URL"),
    api_key=os.getenv("OPENROUTER_API_KEY"),
    is_chat_model=True,
    context_window=8192,
    temperature=1,
)

# OR as_retriever OR as_chat_engine
query_engine = index.as_query_engine(
    # streaming=True,
    llm=llm,
    response_mode="tree_summarize", # OR refine OR compact
)

evaluator = FaithfulnessEvaluator(llm=llm)

answer = query_engine.query("What are llms?")
print("What are llms? \n")
print(answer) # LLMs are a type of technology that enables knowledge generation and reasoning...
print("\n")

eval_result = evaluator.evaluate_response(response=answer)
eval_result.passing

answer = query_engine.query("Why do humans exist?")
print("Why do humans exist? \n")
print(answer) # I don't know
print("\n")

__all__ = ["query_engine"]