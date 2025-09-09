import os
import torch


class Config:
    VECTOR_DB_PATH = "vector_db"
    EMBEDDING_MODEL = "text-embedding-ada-002"
    LLM_MODEL = "gpt-3.5-turbo"
    OPENAI_API_KEY = "sk-" # replace with your openai api key in environment variable also here
    DATA_DIR = "data"
    DEVICE_TYPE = "cuda" if torch.cuda.is_available() else "cpu"
    CHUNK_SIZE = 1000
    CHUNK_OVERLAP = 0

config = Config()