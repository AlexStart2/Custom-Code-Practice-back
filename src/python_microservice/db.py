# db.py
import os
from pymongo import MongoClient
from dotenv import load_dotenv

load_dotenv()
# MONGO_URI = os.getenv("MONGO_URI")
MONGO_URI = "mongodb://localhost:27017"  # Default MongoDB URI, adjust as needed



_client = MongoClient(MONGO_URI)
db      = _client["trainify"]



# Collections:
# - rag_chunks: stores each chunk + embedding + owner
# - rag_jobs: optional, to track batches/jobs
