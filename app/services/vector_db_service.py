from langchain_community.embeddings import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_openai import OpenAI
from typing import List
from app.config.config import config

def add_to_vector_db(chunks: List[str], metadata: dict):
    embeddings = OpenAIEmbeddings(model=config.EMBEDDING_MODEL, api_key=config.OPENAI_API_KEY)
    db = Chroma(persist_directory=config.VECTOR_DB_PATH, embedding_function=embeddings) # Initialize vector database with embeddings function
    db.add_documents(documents=[{"page_content": chunk, **metadata} for chunk in chunks]) # Add documents to the vector database