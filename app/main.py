from langchain_openai import OpenAI
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.responses import JSONResponse
from app.services.document_service import process_document
from app.services.vector_db_service import add_to_vector_db
from app.services.rag_service import qa_chain
from app.config.config import config

from fastapi.encoders import jsonable_encoder
import uvicorn
import os
import shutil
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_openai import OpenAI
from langchain.chains import RetrievalQA

app = FastAPI()

# Initialize vector database 
embeddings = OpenAIEmbeddings(model=config.EMBEDDING_MODEL, api_key=config.OPENAI_API_KEY)
db = Chroma(persist_directory='vector_db', embedding_function=embeddings)

# Initialize LLM and RAG pipeline
llm = OpenAI(model=config.LLM_MODEL, api_key=config.OPENAI_API_KEY)
qa_chain = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=db.as_retriever())

#On startup setup vector db, rag pipeline, openai client
@app.on_event("startup")
async def startup_event():
    """On startup setup vector db, rag pipeline, openai client"""
    # Initialize vector database 
    embeddings = OpenAIEmbeddings(model=config.EMBEDDING_MODEL, api_key=config.OPENAI_API_KEY)
    db = Chroma(persist_directory=config.VECTOR_DB_PATH, embedding_function=embeddings)
    # Initialize LLM and RAG pipeline
    llm = OpenAI(model=config.LLM_MODEL, api_key=config.OPENAI_API_KEY)
    qa_chain = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=db.as_retriever())
    return JSONResponse(content={"message": "Startup successful."})

@app.post("/documents/upload/")
async def upload_document(file: UploadFile = File(...), title: str = Form(...), author: str = Form(...)):
    """Upload and process a document."""
    try:
        #data present in data folder
        file_path = os.path.join(config.DATA_DIR, file.filename)
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        chunks = process_document(file_path) 
        metadata = {"source": file_path, "title": title, "author": author}
        add_to_vector_db(chunks, metadata)
        return JSONResponse(content={"message": "Document processed and added to the database."})
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/documents/")
async def list_documents():
    """List all processed documents."""
    try:
        documents = db.get() #documents present in vector_db folder
        return JSONResponse(content=jsonable_encoder(documents)) #return all documents in json format
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/query/")
async def query(query: str = Form(...)):
    """Perform a RAG query."""
    try:
        result = qa_chain({"query": query})
        return JSONResponse(content={"answer": result["result"]})
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health/")
async def health_check():
    """Check the system's health."""
    return JSONResponse(content={"status": "OK"})

# following will start the server on 8000 port
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)