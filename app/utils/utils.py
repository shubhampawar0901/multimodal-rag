import os
import uuid
import shutil

def generate_unique_filename():
    return str(uuid.uuid4())

def save_file(file, file_path):
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)