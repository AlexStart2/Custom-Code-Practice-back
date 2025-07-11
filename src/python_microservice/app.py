# uvicorn app:app --reload --host 0.0.0.0 --port 5001
# sudo lsof /dev/nvidia-uvm

from fastapi import FastAPI, UploadFile, Request
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
from unstructured.partition.auto import partition
import torch
from sentence_transformers import SentenceTransformer
from langchain.text_splitter import CharacterTextSplitter
from io import BytesIO
import httpx

load_dotenv()
DEVICE   = 'cuda' if torch.cuda.is_available() else 'cpu'
ST_MODEL = SentenceTransformer(
    'intfloat/multilingual-e5-large-instruct',
     device=DEVICE
)
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
ALGORITHM = os.getenv('JWT_ALGORITHM', 'HS256')


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
async def rag_endpoint(files: List[UploadFile], request: Request):
    splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    all_results = []

    for f in files:
        # 1) Read bytes and wrap in BytesIO
        print(f"Processing file: {f.filename}")
        data = await f.read()
        bio = BytesIO(data)

        # 2) Partition into elements
        elems = partition(file=bio)

        for el in elems:
            if hasattr(el, 'text'):
                text = "\n".join(el.text for el in elems)
            else:
                print(el._element_id)



        # 3) Split into manageable chunks
        chunks = splitter.split_text(text)

        # 4) Embed all chunks in one batch
        embeddings = ST_MODEL.encode(
            chunks,
            convert_to_tensor=True,
            normalize_embeddings=True
        )
        embeddings = embeddings.cpu().tolist()  # JSON‑serialize

        # 5) Collect results
        file_results = [
            {'embedding': emb}
            for emb in embeddings
        ]

        all_results.append({'file': f.filename, 'results': file_results})


    # sent the embeddings to the backend to store them in database


    token = request.headers["authorization"].split()[1]

    return await send_to_backend(
        token=token,
        rag_data={ "data": all_results }
    )


async def send_to_backend(token: str, rag_data: dict):
    """
    Function to send embeddings to the backend service.
    This is a placeholder and should be implemented based on your backend API.
    """
    async with httpx.AsyncClient() as client:
        response = await client.post(
            f"{NEST_URL}datasets/store-rag",
            json=rag_data,
            headers={"Authorization": f"Bearer {token}"}
        )
        response.raise_for_status()
        return response.json()