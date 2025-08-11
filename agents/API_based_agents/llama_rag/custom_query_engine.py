# Retrieval-Augmented Generation with LlamaIndex and ChromaDB
# custom RAG query engine

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
from llama_index.core.query_engine import CustomQueryEngine
from llama_index.core.retrievers import BaseRetriever
from llama_index.core import PromptTemplate
from typing import Any
import warnings

warnings.filterwarnings("ignore")
load_dotenv()

qa_prompt = PromptTemplate(
    "Context information is below.\n"
    "---------------------\n"
    "{context_str}\n"
    "---------------------\n"
    "Given the context information and not prior knowledge, "
    "return “I don’t know” unless a high-similarity match exists\n"
    "Query: {query_str}\n"
    "Answer: "
)

llm = OpenAILike(
    model="meta-llama/llama-3-8b-instruct",
    api_base=os.getenv("OPENAI_BASE_URL"),
    api_key=os.getenv("OPENROUTER_API_KEY"),
    is_chat_model=True,
    context_window=8192,
    temperature=1,
)

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

retriever = index.as_retriever(similarity_top_k=5, similarity_threshold=0.4)

class RAGStringQueryEngine(CustomQueryEngine):
    """RAG String Query Engine."""

    retriever: BaseRetriever
    llm: Any
    qa_prompt: PromptTemplate

    def custom_query(self, query_str: str) -> str:
        nodes = self.retriever.retrieve(query_str)

        context_str = "\n\n".join([n.node.get_content() for n in nodes])
        
        prompt = self.qa_prompt.format(context_str=context_str, query_str=query_str)


        resp = self.llm.complete(prompt)
        text = getattr(resp, "text", None) or str(resp)

        return text

query_engine = RAGStringQueryEngine(
    retriever=retriever,
    llm=llm,
    qa_prompt=qa_prompt,
)

answer = query_engine.query("What are llms?")
print("What are llms? \n")
print(answer) # LLMs are a type of technology that enables knowledge generation and reasoning...
print("\n")

answer = query_engine.query("Why do humans exist?")
print("Why do humans exist? \n")
print(answer) # I don't know
print("\n")