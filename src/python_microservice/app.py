# uvicorn app:app --reload --host 0.0.0.0 --port 5001

import os
import subprocess
from typing import List, Dict, Any
from datetime import datetime
import pytz
from fastapi import FastAPI, UploadFile, Form, File, Depends, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from redis import Redis
from rq import Queue
from jose import jwt, JWTError
from bson import ObjectId
from dotenv import load_dotenv

from tasks import process_files_job, update_text_chunk, run_query_history
from db import db


# Load environment variables
load_dotenv()

# Environment configuration
REDIS_URL = os.getenv('REDIS_URL')
if not REDIS_URL:
    raise ValueError("REDIS_URL environment variable is not set")

NEST_URL = os.getenv('NEST_URL')
JWT_SECRET = os.getenv('JWT_SECRET')
if not JWT_SECRET:
    raise ValueError("JWT_SECRET environment variable is not set")

ALGORITHM = os.getenv('JWT_ALGORITHM', 'HS256')

# FastAPI app initialization
app = FastAPI(
    title="RAG Microservice",
    description="Retrieval-Augmented Generation processing service",
    version="1.0.0"
)

# Global variables
available_models: List[str] = []

def fetch_available_models() -> List[str]:
    """
    Fetch available models from Ollama CLI.
    
    Returns:
        List[str]: List of available model names
    """
    try:
        result = subprocess.run(
            ['ollama', 'list'], 
            capture_output=True, 
            text=True,
            timeout=10  # Add timeout for subprocess
        )
        if result.returncode == 0:
            lines = result.stdout.strip().split('\n')
            # Skip header line and extract model names
            models = [line.split()[0] for line in lines[1:] if line.strip()]
            return models
        else:
            print(f"Error fetching models: {result.stderr}")
            return []
    except FileNotFoundError:
        print("Ollama CLI not found. Ensure it is installed and in your PATH.")
        return []
    except subprocess.TimeoutExpired:
        print("Timeout while fetching models from Ollama")
        return []
    except Exception as e:
        print(f"Unexpected error fetching models: {e}")
        return []

# Initialize available models
available_models = fetch_available_models()


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
) -> Dict[str, Any]:
    """
    JWT token verification dependency.
    
    Args:
        creds: HTTP authorization credentials containing the Bearer token
        
    Returns:
        Dict[str, Any]: Decoded token payload
        
    Raises:
        HTTPException: If token is invalid or expired
    """
    if not creds or not creds.credentials:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authorization token is required",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    token = creds.credentials
    try:
        payload = jwt.decode(token, JWT_SECRET, algorithms=[ALGORITHM])
        if not payload.get("id"):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid token payload",
            )
        return payload
    except JWTError as e:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"Could not validate credentials: {str(e)}",
            headers={"WWW-Authenticate": "Bearer"},
        )

def get_user_from_token(token_payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    Get user from database using token payload.
    
    Args:
        token_payload: Decoded JWT token payload
        
    Returns:
        Dict[str, Any]: User document from database
        
    Raises:
        HTTPException: If user not found or invalid token
    """
    user_id = token_payload.get("id")
    if not user_id:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User ID not found in token payload"
        )
    
    try:
        user = db.users.find_one({"_id": ObjectId(user_id)})
        if not user:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="User not found"
            )
        return user
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Database error: {str(e)}"
        )

def validate_model(model: str) -> bool:
    return model in available_models

def validate_object_id(id_string: str, field_name: str = "ID") -> ObjectId:
    try:
        return ObjectId(id_string)
    except Exception:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid {field_name} format"
        )



@app.post("/upload-rag")
async def upload_rag_files(
    datasetName: str = Form(..., description="Name of the dataset"),
    chunk_size: int = Form(..., ge=50, le=2000, description="Size of text chunks"),
    chunk_overlap: int = Form(..., ge=0, le=500, description="Overlap between chunks"),
    files: List[UploadFile] = File(..., description="Files to process"),
    token_payload: Dict[str, Any] = Depends(verify_jwt)
) -> Dict[str, Any]:

    # Validate inputs
    if not datasetName.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Dataset name cannot be empty"
        )
    
    if not files:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="At least one file is required"
        )
    
    if chunk_overlap >= chunk_size:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Chunk overlap must be less than chunk size"
        )

    # Get user from token
    user = get_user_from_token(token_payload)
    user_id = str(user["_id"])

    # Validate file types and sizes
    MAX_FILE_SIZE = 50 * 1024 * 1024  # 50MB


    ALLOWED_EXTENSIONS = [
    '.docx', '.doc', '.odt',
    '.pptx', '.ppt',
    '.xlsx', '.csv', '.tsv',
    '.eml', '.msg',
    '.rtf', '.epub',
    '.html', '.xml',
    '.pdf',
    '.png', '.jpg', '.jpeg', '.heic',
    '.txt', '.md', '.org',
    '.js', '.ts', '.c', '.cpp', '.py', '.java', '.go',
    '.cs', '.rb', '.swift']
    
    for file in files:
        if file.size and file.size > MAX_FILE_SIZE:
            raise HTTPException(
                status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                detail=f"File {file.filename} exceeds maximum size of 50MB"
            )
        if file.filename and not any(file.filename.endswith(ext) for ext in ALLOWED_EXTENSIONS):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"File type {file.filename} is not supported"
            )

    # Create job record in database
    try:
        job_record = {
            "owner": user_id,
            "dataset_name": datasetName.strip(),
            "status": "processing",
            "finishedAt": None,
            "createdAt": datetime.now(pytz.utc),
            "error": None,
            "chunk_size": chunk_size,
            "chunk_overlap": chunk_overlap,
            "file_count": len(files)
        }
        
        job_result = db.jobs_rag.insert_one(job_record)
        job_id = str(job_result.inserted_id)

        # Create file processing records
        file_records = [
            {
                "job_id": job_id,
                "file_name": file.filename,
                "file_size": file.size,
                "content_type": file.content_type,
                "status": "pending",
                "finishedAt": None,
                "createdAt": datetime.now(pytz.utc),
                "error": None,
            } for file in files
        ]
        
        files_result = db.processing_files_rag.insert_many(file_records)
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Database error: {str(e)}"
        )

    # Prepare files for processing
    try:
        files_data = []
        for file in files:
            file_content = await file.read()
            files_data.append({
                'content': file_content,
                'filename': file.filename,
                'content_type': file.content_type,
                'size': file.size
            })
            # Reset file position after reading
            await file.seek(0)

        # Enqueue background job
        job = queue.enqueue(
            process_files_job,
            files_data,
            datasetName.strip(),
            chunk_size,
            chunk_overlap,
            user_id,
            job_id,
            files_result.inserted_ids,
            job_timeout="2h",  # Increased timeout for larger files
        )

        return {
            "success": True,
            "jobId": job.get_id(),
            "status": job.get_status(),
            "message": f"Processing started for {len(files)} files",
            "datasetName": datasetName.strip()
        }
        
    except Exception as e:
        # Update job status to failed
        db.jobs_rag.update_one(
            {"_id": ObjectId(job_id)},
            {"$set": {"status": "failed", "error": str(e)}}
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to start processing: {str(e)}"
        )


@app.patch("/datasets/files/chunks/")
async def update_file_chunk(
    fileId: str = Form(..., description="ID of the file containing the chunk"),
    idx: str = Form(..., description="Index of the chunk to update"),
    text: str = Form(..., description="New text content for the chunk"),
    token_payload: Dict[str, Any] = Depends(verify_jwt)
) -> Dict[str, Any]:
    """
    Update a specific chunk of text in a processed file.
    
    Args:
        fileId: ObjectId string of the processed file
        idx: Index of the chunk to update
        text: New text content for the chunk
        token_payload: JWT token payload
        
    Returns:
        Dict containing success message and updated chunk info
        
    Raises:
        HTTPException: For validation errors or processing failures
    """
    # Validate inputs
    if not fileId.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="File ID is required"
        )
    
    if not text.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Text content cannot be empty"
        )
    
    # Validate chunk index
    try:
        chunk_index = int(idx)
        if chunk_index < 0:
            raise ValueError("Index must be non-negative")
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid chunk index - must be a non-negative integer"
        )

    # Validate ObjectId format
    file_object_id = validate_object_id(fileId, "File ID")
    
    # Get user from token (for authorization)
    user = get_user_from_token(token_payload)
    user_id = str(user["_id"])
    
    # Check if file exists and user has access
    try:
        file_record = db.processed_files.find_one({"_id": file_object_id})
        if not file_record:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="File not found"
            )
        
        # Check if user owns the file (through dataset ownership)
        dataset = db.datasets_rag.find_one({"_id": file_record.get("dataset_id")})
        if not dataset or str(dataset.get("owner")) != user_id:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Access denied - you don't own this file"
            )
            
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Database error while checking file: {str(e)}"
        )
    
    # Validate chunk index against file's chunk count
    chunk_count = file_record.get("chunk_count", 0)
    if chunk_index >= chunk_count:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Chunk index {chunk_index} exceeds file's chunk count ({chunk_count})"
        )
    
    try:
        # Update the chunk
        result = await update_text_chunk(
            file_id=fileId,
            chunk_index=chunk_index,
            text=text.strip(),
        )
        
        return {
            "success": True,
            "message": "Chunk updated successfully",
            "fileId": fileId,
            "chunkIndex": chunk_index,
            "updatedText": text.strip()[:100] + "..." if len(text.strip()) > 100 else text.strip(),
            "timestamp": datetime.now(pytz.utc).isoformat(),
            "result": result
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error updating chunk: {str(e)}"
        )





@app.post("/models/rag-query-history")
async def process_rag_query(
    datasetId: str = Form(..., description="ID of the dataset to query"),
    query: str = Form(..., description="Query text for RAG processing"),
    model: str = Form(..., description="Model to use for query processing"),
    token_payload: Dict[str, Any] = Depends(verify_jwt),
) -> Dict[str, Any]:
    """
    Process a RAG query against a specific dataset with query history tracking.
    
    Args:
        datasetId: ObjectId string of the dataset to query
        query: The question or query to process
        model: Name of the model to use for processing
        token_payload: JWT token payload
        
    Returns:
        Dict containing the query response and metadata
        
    Raises:
        HTTPException: For validation errors or processing failures
    """
    # Validate inputs
    if not datasetId.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Dataset ID is required"
        )
    
    if not query.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Query text cannot be empty"
        )
    
    if not model.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Model name is required"
        )

    # Get user from token
    user = get_user_from_token(token_payload)
    user_id = str(user["_id"])

    # Validate ObjectId format
    dataset_object_id = validate_object_id(datasetId, "Dataset ID")

    # Check if dataset exists and user has access
    try:
        dataset = db.datasets_rag.find_one({"_id": dataset_object_id})
        if not dataset:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Dataset not found"
            )
        
        # Check dataset ownership
        if str(dataset.get("owner")) != user_id:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Access denied - you don't own this dataset"
            )
            
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Database error while checking dataset: {str(e)}"
        )
    
    # Validate the model
    if not validate_model(model):
        # Refresh available models in case of new installations
        global available_models
        available_models = fetch_available_models()
        
        if not validate_model(model):
            available_models_str = ", ".join(available_models) if available_models else "None"
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Model '{model}' is not available. Available models: {available_models_str}"
            )

    try:
        # Process the query
        response = await run_query_history(
            query=query.strip(),
            model=model.strip(),
            dataset_id=datasetId,
            user_id=user_id
        )
        
        # Add metadata to response
        if isinstance(response, dict):
            response.update({
                "metadata": {
                    "dataset_id": datasetId,
                    "dataset_name": dataset.get("name", "Unknown"),
                    "model_used": model,
                    "query_timestamp": datetime.now(pytz.utc).isoformat(),
                    "user_id": user_id
                }
            })
        
        return response
        
    except Exception as e:
        # Log the error for debugging
        print(f"RAG query processing error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error processing query: {str(e)}"
        )