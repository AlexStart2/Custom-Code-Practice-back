# uvicorn app:app --reload --host 0.0.0.0 --port 5001
# sudo lsof /dev/nvidia-uvm

from fastapi import FastAPI, UploadFile, Form, File    
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi import Depends, HTTPException, status
from typing import List
from jose import jwt, JWTError
# from redis import Redis   # this will be used to obtain concurent jobs for multiple requests
# from rq import Queue
# from rq.job import Job
import os
from dotenv import load_dotenv
from tasks import process_and_store, run_query

from db import db


load_dotenv()

NEST_URL = os.getenv('NEST_URL')

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=['http://localhost:5173'],  # your React app’s origin
    allow_methods=['GET', 'POST', 'PUT', 'DELETE', 'OPTIONS'],
    allow_headers=['*'],
    allow_credentials=True,
)

# db = Redis.from_url(REDIS_URL)
# queue = Queue('rag-jobs', connection=db)

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
    files: List[UploadFile] = File(...),
    token_payload: dict = Depends(verify_jwt)
):
    # sent the embeddings to the backend to store them in database
    user_id = token_payload.get("id") 
    return await process_and_store(files, dataset_name=datasetName, user_id=user_id)



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

    response = await run_query(
        query=query,
        model=model,
        dataset_id=datasetId,
    )

    # For now, we just return a placeholder response
    return response



@app.get("/health/db")
async def check_db():
    try:
        listy = db.list_collection_names()
        print(listy)
        return {"status": "ok", "db": "reachable"}
    except Exception as e:
        # If anything goes wrong, return a 503 with the error
        raise HTTPException(status_code=503, detail=f"DB connection failed: {e}")