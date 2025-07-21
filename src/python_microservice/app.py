# uvicorn app:app --reload --host 0.0.0.0 --port 5001

from fastapi import FastAPI, UploadFile, Form, File    
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi import Depends, HTTPException, status
from typing import List
from jose import jwt, JWTError
from redis import Redis   # this will be used to obtain concurent jobs for multiple requests
from rq import Queue
import os
from dotenv import load_dotenv
from tasks import process_files_job, run_query, update_text_chunk
import pytz
from datetime import datetime
from db import db
from bson import ObjectId


load_dotenv()

REDIS_URL = os.getenv('REDIS_URL')
if not REDIS_URL:
    raise ValueError("REDIS_URL environment variable is not set")

NEST_URL = os.getenv('NEST_URL')

app = FastAPI()

models = []

# get the list of models from the ollama list command
try:
    import subprocess
    result = subprocess.run(['ollama', 'list'], capture_output=True, text=True)
    if result.returncode == 0:
        models = [line.split()[0] for line in result.stdout.strip().split('\n') if line]
    else:
        print(f"Error fetching models: {result.stderr}")
except FileNotFoundError:
    print("Ollama CLI not found. Ensure it is installed and in your PATH.")


app.add_middleware(
    CORSMiddleware,
    allow_origins=['http://localhost:5173', REDIS_URL],  # your React app’s origin
    allow_methods=['GET', 'POST', 'PUT', 'DELETE', 'OPTIONS', 'PATCH'],
    allow_headers=['*'],
    allow_credentials=True,
)

redis_conn = Redis.from_url(REDIS_URL)
queue      = Queue("rag-jobs", connection=redis_conn)

bearer_scheme = HTTPBearer()
JWT_SECRET = os.getenv('JWT_SECRET')
ALGORITHM = os.getenv('JWT_ALGORITHM')

def verify_jwt(
    creds: HTTPAuthorizationCredentials = Depends(bearer_scheme)
):
    """
    Dependency that:
      - Extracts the Bearer token
      - Decodes + verifies signature & expiration
      - Raises 401 if invalid
      - Returns the token payload (claims) on success
    """
    token = creds.credentials
    try:
        payload = jwt.decode(token, JWT_SECRET, algorithms=[ALGORITHM])
    except JWTError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Could not validate credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return payload



@app.post("/upload-rag", dependencies=[Depends(verify_jwt)])
async def rag_endpoint(
    datasetName: str = Form(...),
    chunk_size: int = Form(...),
    chunk_overlap: int = Form(...),
    files: List[UploadFile] = File(...),
    token_payload: dict = Depends(verify_jwt)
):
    # sent the embeddings to the backend to store them in database
    try:
        user_id = token_payload.get("id") 
    except KeyError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User ID not found in token payload"
        )
    if not datasetName or not files:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="dataset name and files are required"
        )

    user = db.users.find_one({"_id": ObjectId(user_id)})
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found"
        )

    
    if chunk_size <= 10 or chunk_overlap < 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid chunk size or overlap"
        )

    # put in data base new job

    try:
        r = db.jobs_rag.insert_one({
            "owner": user_id,
            "dataset_name": datasetName,
            "status": "processing",
            "finishedAt": None,
            "createdAt": datetime.now(pytz.utc),
            "error": None,
        })

        job_id = str(r.inserted_id) 

        # put in data base files
        files_job_id = db.processing_files_rag.insert_many([
            {
                "job_id": job_id,
                "file_name": file.filename,
                "status": "pending",
                "finishedAt": None,
                "createdAt": datetime.now(pytz.utc),
                "error": None,
            } for file in files
        ])
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error inserting job into database: {str(e)}"
        )


    files_data = []
    for file in files:
        file_content = await file.read()
        files_data.append({
            'content': file_content,
            'filename': file.filename,
            'content_type': file.content_type,
            'size': file.size
        })

    # Enqueue the job; it returns an RQ Job instance
    job = queue.enqueue(
        process_files_job,
        files_data,
        datasetName,
        chunk_size,
        chunk_overlap,
        user_id,
        job_id,
        files_job_id.inserted_ids,
        job_timeout="1h",   # adjust
    )

    return {"jobId": job.get_id(), "status": job.get_status()}  # "queued"


@app.post("/models/rag-query", dependencies=[Depends(verify_jwt)])
async def rag_query_endpoint(
    # recieve a json with the query, datasetId, and model
    datasetId: str = Form(...),
    query: str = Form(...),
    model: str = Form(...),
    # optional historyId for conversation context
    historyId: str = Form(None)
):
    """
    Endpoint to handle RAG queries.
    """

    if not datasetId or not query or not model:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="dataset, query, and model are required"
        )

    dataset = db.datasets_rag.find_one({"_id": ObjectId(datasetId)})
    if not dataset:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Dataset not found"
        )
    
    # Validate the model
    if model not in models[1:]:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Model '{model}' is not available"
        )


    try:
        response = await run_query(
            query=query,
            model=model,
            dataset_id=datasetId,
            historyId=historyId

        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error processing query: {str(e)}"
        )

    # For now, we just return a placeholder response
    return response


@app.patch("/datasets/files/chunks/", dependencies=[Depends(verify_jwt)])
async def update_chunk(
    fileId: str = Form(...),
    idx: str = Form(...),
    text: str = Form(...)
):
    """
    Endpoint to update a chunk of a file in a dataset.
    """

    if not fileId or not idx or not text:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="file, idx, and text are required"
        )
    
    file = db.processed_files.find_one({"_id": ObjectId(fileId)})
    if not file:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="File not found"
        )
    
    try:
        # Call the actual update_chunk function from tasks.py
        result = await update_text_chunk(
            file_id=fileId,
            chunk_index=int(idx),
            text=text,
        )
        
        return {
            "message": "Chunk updated successfully",
            "fileId": fileId,
            "chunkIndex": idx,
            "text": text,
            "result": result
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error updating chunk: {str(e)}"
        )

