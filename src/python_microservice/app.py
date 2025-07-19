# uvicorn app:app --reload --host 0.0.0.0 --port 5001
# sudo lsof /dev/nvidia-uvm

from fastapi import FastAPI, UploadFile, Form, File    
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi import Depends, HTTPException, status
from typing import List
from jose import jwt, JWTError
from redis import Redis   # this will be used to obtain concurent jobs for multiple requests
from rq import Queue
from rq.job import Job
import os
from dotenv import load_dotenv
from tasks import process_files_job, run_query
import pytz
from datetime import datetime

from db import db


load_dotenv()

REDIS_URL = os.getenv('REDIS_URL')
if not REDIS_URL:
    raise ValueError("REDIS_URL environment variable is not set")

NEST_URL = os.getenv('NEST_URL')

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=['http://localhost:5173', REDIS_URL],  # your React app’s origin
    allow_methods=['GET', 'POST', 'PUT', 'DELETE', 'OPTIONS'],
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
    
        # put in data base new job
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
    model: str = Form(...)
):
    """
    Endpoint to handle RAG queries.
    """

    try:
        response = await run_query(
            query=query,
            model=model,
            dataset_id=datasetId,
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error processing query: {str(e)}"
        )

    # For now, we just return a placeholder response
    return response


@app.get("/jobs/{job_id}", dependencies=[Depends(verify_jwt)])          # ???????????
async def get_job_status(job_id: str):
    job = Job.fetch(job_id, connection=redis_conn)
    return {"id": job.id, "status": job.get_status(), "result": job.result}

@app.post("/jobs/{job_id}/cancel", dependencies=[Depends(verify_jwt)])
async def cancel_job(job_id: str):
    job = Job.fetch(job_id, connection=redis_conn)
    job.cancel()   # will raise if already started
    return {"id": job.id, "status": "canceled"}

