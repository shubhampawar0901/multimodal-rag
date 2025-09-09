from langchain_community.document_loaders import UnstructuredFileLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from typing import List
from app.config.config import config

def process_document(file_path: str) -> List[str]:
    loader = UnstructuredFileLoader(file_path)
    docs = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=config.CHUNK_SIZE, chunk_overlap=config.CHUNK_OVERLAP) 
    chunks = text_splitter.split_documents(docs)
    return [chunk.page_content for chunk in chunks]