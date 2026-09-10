import os
import uuid
import jwt
import shutil
import chromadb

from datetime import datetime, timedelta, timezone
from pathlib import Path

from fastapi import FastAPI, HTTPException, Depends, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel, EmailStr, Field
from pwdlib import PasswordHash
from sqlalchemy import String, create_engine, select
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column
from dotenv import load_dotenv

from loaders.pdf_loader_and_extractor import load_pdf
from loaders.txt_loader import load_txt
from loaders.docx_loader_and_extractor import load_docx
from loaders.xlsx_loader_and_extractor import load_xlsx
from loaders.img_loader_and_extractor import load_image

from chunker import chunk_documents
from embeddings import create_embeddings
from vector_db import create_vector_db
from retriever import run_rag


# =========================================================
# CONFIG
# =========================================================

load_dotenv()

app = FastAPI()


# =========================================================
# CORS
# =========================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

password_hash = PasswordHash.recommended()
security = HTTPBearer()

JWT_SECRET_KEY = os.getenv("JWT_SECRET_KEY")
JWT_ALGORITHM = "HS256"
JWT_EXPIRATION_HOURS = 24

UPLOAD_DIR = Path("user_uploads")
CHROMA_PATH = r"E:\Machine_learning\AI\RAG\PROJECT\chroma_db"

UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

if not JWT_SECRET_KEY:
    raise RuntimeError("JWT_SECRET_KEY is missing from .env")


# =========================================================
# DATABASE
# =========================================================

class Base(DeclarativeBase):
    pass


class User(Base):
    __tablename__ = "users"

    id: Mapped[str] = mapped_column(String, primary_key=True)
    email: Mapped[str] = mapped_column(String, unique=True, index=True)
    password_hash: Mapped[str] = mapped_column(String)


class Document(Base):
    __tablename__ = "documents"

    id: Mapped[str] = mapped_column(String, primary_key=True)
    user_id: Mapped[str] = mapped_column(String, index=True)
    filename: Mapped[str] = mapped_column(String)
    file_type: Mapped[str] = mapped_column(String)
    status: Mapped[str] = mapped_column(String)
    uploaded_at: Mapped[str] = mapped_column(String)


engine = create_engine("sqlite:///users.db", connect_args={"check_same_thread": False})

Base.metadata.create_all(engine)


# =========================================================
# REQUEST / RESPONSE MODELS
# =========================================================

class UserRegister(BaseModel):
    email: EmailStr
    password: str = Field(min_length=8)


class UserLogin(BaseModel):
    email: EmailStr
    password: str


class UserResponse(BaseModel):
    id: str
    email: EmailStr


class TokenResponse(BaseModel):
    access_token: str
    token_type: str


class ChatRequest(BaseModel):
    query: str = Field(min_length=1)


# =========================================================
# JWT
# =========================================================

def create_access_token(user_id):

    expiration = datetime.now(timezone.utc) + timedelta(hours=JWT_EXPIRATION_HOURS)

    payload = {
        "sub": user_id,
        "exp": expiration
    }

    token = jwt.encode(payload, JWT_SECRET_KEY, algorithm=JWT_ALGORITHM)

    return token


def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)):

    token = credentials.credentials

    try:
        payload = jwt.decode(token, JWT_SECRET_KEY, algorithms=[JWT_ALGORITHM])

        user_id = payload.get("sub")

        if not user_id:
            raise HTTPException(status_code=401, detail="Invalid token.")

    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Token has expired.")

    except jwt.InvalidTokenError:
        raise HTTPException(status_code=401, detail="Invalid token.")

    with Session(engine) as session:

        user = session.get(User, user_id)

        if not user:
            raise HTTPException(status_code=401, detail="User not found.")

        return {
            "id": user.id,
            "email": user.email
        }


# =========================================================
# ROOT
# =========================================================

@app.get("/")
def root():

    return {
        "message": "RAG API is running"
    }


# =========================================================
# REGISTER
# =========================================================

@app.post("/register", response_model=UserResponse)
def register(user_data: UserRegister):

    email = user_data.email.lower()

    with Session(engine) as session:

        existing_user = session.scalar(select(User).where(User.email == email))

        if existing_user:
            raise HTTPException(status_code=400, detail="User with this email already exists.")

        new_user = User(
            id=str(uuid.uuid4()),
            email=email,
            password_hash=password_hash.hash(user_data.password)
        )

        session.add(new_user)
        session.commit()
        session.refresh(new_user)

        return UserResponse(id=new_user.id, email=new_user.email)


# =========================================================
# LOGIN
# =========================================================

@app.post("/login", response_model=TokenResponse)
def login(user_data: UserLogin):

    email = user_data.email.lower()

    with Session(engine) as session:

        user = session.scalar(select(User).where(User.email == email))

        if not user:
            raise HTTPException(status_code=401, detail="Invalid email or password.")

        valid_password = password_hash.verify(user_data.password, user.password_hash)

        if not valid_password:
            raise HTTPException(status_code=401, detail="Invalid email or password.")

        token = create_access_token(user.id)

        return TokenResponse(access_token=token, token_type="bearer")


# =========================================================
# CURRENT USER
# =========================================================

@app.get("/me")
def get_me(current_user=Depends(get_current_user)):

    return current_user


# =========================================================
# FILE LOADER
# =========================================================

def load_uploaded_file(file_path):

    extension = Path(file_path).suffix.lower()

    if extension == ".pdf":
        return load_pdf(file_path)

    if extension == ".txt":
        return load_txt(file_path)

    if extension == ".docx":
        return load_docx(file_path)

    if extension == ".xlsx":
        return load_xlsx(file_path)

    if extension in [".png", ".jpg", ".jpeg"]:
        return load_image(file_path)

    raise ValueError("Unsupported file type.")


# =========================================================
# UPLOAD DOCUMENT
# =========================================================

@app.post("/upload")
def upload_document(file: UploadFile = File(...), current_user=Depends(get_current_user)):

    allowed_extensions = {
        ".pdf",
        ".txt",
        ".docx",
        ".xlsx",
        ".png",
        ".jpg",
        ".jpeg"
    }

    extension = Path(file.filename).suffix.lower()

    if extension not in allowed_extensions:
        raise HTTPException(status_code=400, detail="Unsupported file type.")

    user_id = current_user["id"]
    document_id = str(uuid.uuid4())
    uploaded_at = datetime.now(timezone.utc).isoformat()

    user_folder = UPLOAD_DIR / user_id / document_id
    user_folder.mkdir(parents=True, exist_ok=True)

    safe_filename = Path(file.filename).name
    file_path = user_folder / safe_filename

    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    try:
        documents = load_uploaded_file(str(file_path))

        for document in documents:
            document["metadata"]["user_id"] = user_id
            document["metadata"]["document_id"] = document_id

        chunks = chunk_documents(documents)
        chunks = create_embeddings(chunks)

        create_vector_db(chunks)

    except Exception as e:

        if user_folder.exists():
            shutil.rmtree(user_folder)

        raise HTTPException(status_code=500, detail=f"Error processing document: {str(e)}")

    with Session(engine) as session:

        new_document = Document(
            id=document_id,
            user_id=user_id,
            filename=safe_filename,
            file_type=extension.replace(".", ""),
            status="ready",
            uploaded_at=uploaded_at
        )

        session.add(new_document)
        session.commit()

    return {
        "message": "Document uploaded and processed successfully.",
        "document_id": document_id,
        "filename": safe_filename,
        "user_id": user_id,
        "extracted_documents": len(documents),
        "chunks_created": len(chunks)
    }


# =========================================================
# GET USER DOCUMENTS
# =========================================================

@app.get("/documents")
def get_documents(current_user=Depends(get_current_user)):

    user_id = current_user["id"]

    with Session(engine) as session:

        documents = session.scalars(
            select(Document).where(Document.user_id == user_id)
        ).all()

        return [
            {
                "document_id": document.id,
                "filename": document.filename,
                "file_type": document.file_type,
                "status": document.status,
                "uploaded_at": document.uploaded_at
            }
            for document in documents
        ]


# =========================================================
# DELETE DOCUMENT
# =========================================================

@app.delete("/documents/{document_id}")
def delete_document(document_id: str, current_user=Depends(get_current_user)):

    user_id = current_user["id"]

    with Session(engine) as session:

        document = session.scalar(
            select(Document).where(
                Document.id == document_id,
                Document.user_id == user_id
            )
        )

        if not document:
            raise HTTPException(status_code=404, detail="Document not found.")

        chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)
        collection = chroma_client.get_collection(name="multi_documents")

        collection.delete(
            where={
                "$and": [
                    {"user_id": user_id},
                    {"document_id": document_id}
                ]
            }
        )

        user_folder = UPLOAD_DIR / user_id / document_id

        if user_folder.exists():
            shutil.rmtree(user_folder)

        session.delete(document)
        session.commit()

    return {
        "message": "Document deleted successfully.",
        "document_id": document_id
    }


# =========================================================
# CHAT
# =========================================================

@app.post("/chat")
def chat(chat_data: ChatRequest, current_user=Depends(get_current_user)):

    user_id = current_user["id"]

    result = run_rag(chat_data.query, user_id)

    return result