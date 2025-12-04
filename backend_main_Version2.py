from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
from sqlalchemy import create_engine, Column, Integer, String, DateTime, Text, ForeignKey
from sqlalchemy.orm import sessionmaker, declarative_base, relationship
from datetime import datetime
import os
import shutil

# ----- CONFIG -----
DATABASE_URL = "sqlite:///./rag_chatbot.db"
UPLOAD_DIR = "./uploaded_documents"
if not os.path.exists(UPLOAD_DIR):
    os.makedirs(UPLOAD_DIR)

# ----- DB SETUP -----
Base = declarative_base()
engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


class Document(Base):
    __tablename__ = "documents"
    id = Column(Integer, primary_key=True, index=True)
    filename = Column(String, unique=True)
    filepath = Column(String)
    filetype = Column(String)
    uploaded_at = Column(DateTime, default=datetime.utcnow)
    chats = relationship("Chat", back_populates="document")


class Chat(Base):
    __tablename__ = "chats"
    id = Column(Integer, primary_key=True, index=True)
    document_id = Column(Integer, ForeignKey("documents.id"))
    user_message = Column(Text)
    assistant_message = Column(Text)
    timestamp = Column(DateTime, default=datetime.utcnow)
    document = relationship("Document", back_populates="chats")


Base.metadata.create_all(bind=engine)

# ----- FASTAPI APP -----
app = FastAPI(
    title="RAG-Chatbot Backend",
    description="Backend for document upload and chat with RAG/LLM integration.",
    version="1.0"
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Change for production!
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ----- MODELS -----
class DocumentOut(BaseModel):
    id: int
    filename: str
    filetype: str
    uploaded_at: datetime

    class Config:
        orm_mode = True

class ChatRequest(BaseModel):
    message: str

class ChatResponse(BaseModel):
    user_message: str
    assistant_message: str
    timestamp: datetime

    class Config:
        orm_mode = True

# ----- ENDPOINTS -----
@app.post("/documents/", response_model=DocumentOut)
async def upload_document(file: UploadFile = File(...)):
    session = SessionLocal()
    filepath = os.path.join(UPLOAD_DIR, file.filename)
    
    if session.query(Document).filter_by(filename=file.filename).first():
        session.close()
        raise HTTPException(status_code=400, detail="Document with this filename already exists.")
    
    with open(filepath, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    doc = Document(
        filename=file.filename,
        filepath=filepath,
        filetype=file.content_type
    )
    session.add(doc)
    session.commit()
    session.refresh(doc)
    session.close()
    return doc

@app.get("/documents/", response_model=list[DocumentOut])
def list_documents():
    session = SessionLocal()
    docs = session.query(Document).all()
    session.close()
    return docs

@app.delete("/documents/{doc_id}/")
def delete_document(doc_id: int):
    session = SessionLocal()
    doc = session.query(Document).filter_by(id=doc_id).first()
    if not doc:
        session.close()
        raise HTTPException(status_code=404, detail="Document not found")
    if os.path.exists(doc.filepath):
        os.remove(doc.filepath)
    session.delete(doc)
    session.commit()
    session.close()
    return {"detail": "Document deleted"}

@app.post("/chat/{doc_id}/", response_model=ChatResponse)
def chat_with_document(doc_id: int, chat_req: ChatRequest):
    session = SessionLocal()
    doc = session.query(Document).filter_by(id=doc_id).first()
    if not doc:
        session.close()
        raise HTTPException(status_code=404, detail="Document not found")
    
    user_message = chat_req.message
    # --- DUMMY RAG/LLM CHATBOT LOGIC ---
    # Here you would implement real retrieval augmented generation over the document
    # For demo, just echo document filename and user's question.
    answer = f"(Demo Answer) You asked: '{user_message}' about document '{doc.filename}'. [Integrate real RAG/LLM here]"
    
    chat = Chat(
        document_id=doc_id,
        user_message=user_message,
        assistant_message=answer
    )
    session.add(chat)
    session.commit()
    session.refresh(chat)
    session.close()
    return chat

@app.get("/chat/{doc_id}/", response_model=list[ChatResponse])
def get_chat_history(doc_id: int):
    session = SessionLocal()
    chats = session.query(Chat).filter_by(document_id=doc_id).order_by(Chat.timestamp).all()
    session.close()
    return chats

@app.get("/documents/{doc_id}/file/")
def get_document_file(doc_id: int):
    session = SessionLocal()
    doc = session.query(Document).filter_by(id=doc_id).first()
    session.close()
    if not doc or not os.path.exists(doc.filepath):
        raise HTTPException(status_code=404, detail="File not found")
    return FileResponse(doc.filepath, media_type=doc.filetype, filename=doc.filename)