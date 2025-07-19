from unstructured.partition.auto import partition
import torch
from sentence_transformers import SentenceTransformer
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.embeddings import Embeddings
from io import BytesIO
from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_chroma import Chroma
from langchain.chains import create_retrieval_chain
from langchain_core.documents import Document       
from typing import List
from fastapi import UploadFile
from db import db
from datetime import datetime
import pytz
from bson.objectid import ObjectId
from fastapi import HTTPException



DEVICE   = 'cuda' if torch.cuda.is_available() else 'cpu'

ST_MODEL = SentenceTransformer(
    'intfloat/multilingual-e5-large-instruct',
    device=DEVICE
)



class SentenceTransformerEmbeddings(Embeddings):
    """
    Wrapper class to make SentenceTransformer compatible with LangChain's Embeddings interface.
    """
    def __init__(self, model: SentenceTransformer):
        self.model = model
    

def rag_chain(model: str = "mistral", vector_store: Chroma = None):

    model = ChatOllama(model=model)


    prompt = PromptTemplate.from_template(
        """
        <s> [Instructions] You are a friendly assistant. Answer the question based only on the following context. 
        If you don't know the answer, then reply, No Context available for this question {input}. [/Instructions] </s> 
        [Instructions] Question: {input} 
        Context: {context} 
        """
    )

    #Create chain
    retriever = vector_store.as_retriever(
        search_type="similarity_score_threshold",
        search_kwargs={
            "k": 1,
            "score_threshold": 0.6,
        },
    )

    document_chain = create_stuff_documents_chain(model, prompt)
    chain = create_retrieval_chain(retriever, document_chain)
    
    #
    return chain


def ask(query: str, model: str, vector_store: Chroma):
    chain = rag_chain(model=model, vector_store=vector_store)
    # invoke chain

    try:
        result = chain.invoke({"input": query})
        return result
    except Exception as e:
        return HTTPException(
            status_code=501,
            detail=f"Error running query: {str(e)}"
        )


def embed_in_batches(texts: list[str], batch_size: int = 16) -> list[list[float]]:
    all_embs = []
    for i in range(0, len(texts), batch_size):

        # Process texts in batches
        if i + batch_size > len(texts):
            batch = texts[i:]
        else:
            batch = texts[i : i + batch_size]
        
        embs = ST_MODEL.encode(
            batch,
            convert_to_tensor=True,
            normalize_embeddings=True,
        )
        embs = embs.cpu().tolist()
        all_embs.extend(embs)
        # free GPU memory between batches
        if torch.cuda.is_available():
            del embs
            torch.cuda.empty_cache()
    return all_embs


def process_and_store_file(
    file: UploadFile,
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
) -> dict:
    result = None

    elems = partition(file=BytesIO(file['content']), extract_images_in_pdf=True, encoding='utf-8')

    text = ""

    text_segments = [
        el.text
        for el in elems
        if getattr(el, 'text', None)   # attribute exists
        and isinstance(el.text, str)    # it really is a string
        and el.text.strip()             # non‐empty after stripping
    ]
    text = "\n".join(text_segments)


    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)

    chunks = splitter.split_text(text)

    try:
        embeddings = embed_in_batches(chunks)
    except Exception as e:
        raise ValueError(f"Error embedding chunks from file {file['filename']}: {e}")

    # 7) Collect results

    result = {
        'file': file['filename'],
        'results': [
            {'text': c, 'embedding': e} 
            for c, e in zip(chunks, embeddings)
        ]
    }

    return result


async def process_files_job(
    files_data: List[bytes],
    dataset_name: str,
    chunk_size: int,
    chunk_overlap: int,
    user_id: str,
    job_id: str,
    files_job_id: List[str],
):
    """
    Synchronous function called by RQ worker.
    - files_data: list of raw bytes
    - files_meta: list of {filename: str}
    """
    # Reconstruct UploadFile-like objects if needed, or change
    # your process_and_store to accept raw bytes+meta instead of UploadFile.

    results = []

    for file, file_id in zip(files_data, files_job_id):

        try:
            # 1) Process each file
            db.processing_files_rag.update_one(
                {"_id": file_id},
                {"$set": {"status": "processing"}},
            )

            # call your existing logic file‐by‐file

            result = process_and_store_file(
                file=file,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
            )

            results.append(result)

            # update file status in database
            db.processing_files_rag.update_one(
                {"_id": file_id},
                {"$set": {"status": "completed", "finishedAt": datetime.now(pytz.utc)}},
            )
        except Exception as e:
            # update job status in database
            db.processing_files_rag.update_one(
                {"_id": ObjectId(file_id)},
                {
                    "$set": {
                        "status": "failed",
                        "error": str(e),
                        "finishedAt": datetime.now(pytz.utc)
                    }
                }
            )
            
            continue

        
    complete_one = False
    file_statuses = db.processing_files_rag.find(
        {"job_id": job_id}
    )
    statuses = {f["status"] for f in file_statuses}

    if statuses == {"completed"}:
        final_status = "completed"
        complete_one = True
    elif "processing" in statuses or "pending" in statuses:
        final_status = "failed"
    elif "failed" in statuses and "completed" in statuses:
        final_status = "partial"   # some succeeded, some failed
        complete_one = True
    else:
        final_status = "failed"


    db.jobs_rag.update_one(
        {"_id": ObjectId(job_id)},
        {
            "$set": {
                "status": final_status,
                # "finishedAt": datetime.now(pytz.utc)
            }
        },
    )

    files_ids = db.processed_files.insert_many([
        {
            # "job_id": job_id,
            "file_name": result['file'],
            "results": result['results'],
        } for result in results
    ]).inserted_ids


    # 8) Store all results in the database
    if complete_one:
            db.datasets_rag.insert_one({
                "owner": user_id,
                "name": dataset_name,
                "files": files_ids,
                "createdAt": datetime.now(pytz.utc),
            })


    return {"jobId": job_id, "status": final_status}

    
async def run_query(
        dataset_id: str,
        query: str,
        model: str = "mistral"
):
    try:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()  # clear GPU memory

        dataset = db.datasets_rag.find_one({"_id": ObjectId(dataset_id)})

        if not dataset:
            raise ValueError(f"Dataset with ID {dataset_id} not found")
                
        # Extract texts and embeddings from stored data
        texts = []
        embeddings = []
        metadatas = []
        
        for file_result in dataset["chunks"]:
            filename = file_result["file"]
            for chunk_data in file_result["results"]:
                texts.append(chunk_data["text"])
                embeddings.append(chunk_data["embedding"])
                metadatas.append({"source": filename})
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error retrieving dataset: {str(e)}"
        )

    try:

        # Create Chroma vector store with Document objects
        vector_store = Chroma(
            embedding_function=SentenceTransformerEmbeddings(ST_MODEL)
        )

        vector_store.add_texts(
            texts=texts,
            metadatas=metadatas,
            embeddings=embeddings
        )
    except Exception as e:
        raise HTTPException(
            status_code=501,
            detail=f"Error creating vector store: {str(e)}"
        )

    try:
        answer = ask(query=query, model=model, vector_store=vector_store)
    except Exception as e:
        return HTTPException(
            500, f"Error running query: {str(e)}"
        )

    return answer