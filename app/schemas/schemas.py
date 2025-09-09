from pydantic import BaseModel

class Document(BaseModel):
    id: str
    title: str
    author: str
    source: str
    content: str