from langchain_chroma.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_community.llms import OpenAI
from app.config.config import config
from app.services.vector_db_service import add_to_vector_db
from app.services.document_service import process_document
from app.utils.utils import generate_unique_filename, save_file
from fastapi import UploadFile
from fastapi.responses import JSONResponse
from fastapi.encoders import jsonable_encoder
from fastapi import HTTPException
from typing import List

def qa_chain(query: str):
    llm = OpenAI(model=config.LLM_MODEL, api_key=config.OPENAI_API_KEY)
    db = Chroma(persist_directory=config.VECTOR_DB_PATH) # Initialize vector database with embeddings function
    qa_chain = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=db.as_retriever())
    return qa_chain({"query": query})