import shutil
import os
from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
from ingest import ingest, setup_collection
from chain import ask

app = FastAPI(title='RAG Q&A API')

# Folder where uploaded files are temporarily saved on the server
UPLOAD_DIR = 'uploads'
os.makedirs(UPLOAD_DIR, exist_ok=True)


class QueryRequest(BaseModel):
    question: str
    collection: str = 'docs'


@app.get('/')
def root():
    return {'status': 'RAG API is running'}


@app.post('/upload')
async def upload_document(file: UploadFile = File(...)):
    """
    Accepts a PDF, DOCX, or TXT file from the frontend.
    Saves it to disk, runs the full ingestion pipeline on it,
    then returns a confirmation message.
    """
    # Validate file type before doing any work
    allowed = ('.pdf', '.docx', '.txt')
    if not file.filename.endswith(allowed):
        raise HTTPException(status_code=400, detail=f'Unsupported file type. Allowed: {allowed}')

    # Save the uploaded file to disk
    save_path = os.path.join(UPLOAD_DIR, file.filename)
    with open(save_path, 'wb') as f:
        shutil.copyfileobj(file.file, f)

    # Set up a fresh Qdrant collection and ingest the document
    setup_collection('docs')
    ingest(save_path)

    return {'status': 'ingested', 'filename': file.filename}


@app.post('/ask')
async def query(request: QueryRequest):
    """
    Accepts a question and collection name.
    Runs the full RAG pipeline: retrieve → confidence check → LLM generation.
    Returns the answer, confidence info, and source citations.
    """
    try:
        result = ask(request.question, request.collection)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# Run with: uvicorn api:app --reload --port 8000
