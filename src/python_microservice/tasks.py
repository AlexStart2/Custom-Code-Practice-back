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



DEVICE   = 'cuda' if torch.cuda.is_available() else 'cpu'





class SentenceTransformerEmbeddings(Embeddings):
    """
    Wrapper class to make SentenceTransformer compatible with LangChain's Embeddings interface.
    """
    def __init__(self, model: SentenceTransformer):
        self.model = model
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed a list of documents."""
        embeddings = self.model.encode(
            texts,
            convert_to_tensor=True,
            normalize_embeddings=True
        )
        return embeddings.cpu().tolist()
    
    def embed_query(self, text: str) -> List[float]:
        """Embed a single query."""
        embedding = self.model.encode(
            [text],
            convert_to_tensor=True,
            normalize_embeddings=True
        )
        return embedding.cpu().tolist()[0]
    
ST_MODEL = SentenceTransformer(
    'intfloat/multilingual-e5-large-instruct',
    device=DEVICE
)



def rag_chain(model: str = "mistral", vector_store: Chroma = None):
    model = ChatOllama(model=model)
    #
    prompt = PromptTemplate.from_template(
        """
        <s> [Instructions] You are a friendly assistant. Answer the question based only on the following context. 
        If you don't know the answer, then reply, No Context availabel for this question {input}. [/Instructions] </s> 
        [Instructions] Question: {input} 
        Context: {context} 
        Answer: [/Instructions]
        """
    )

    #Create chain
    retriever = vector_store.as_retriever(
        search_type="similarity_score_threshold",
        search_kwargs={
            "k": 1,
            "score_threshold": 0.7,
        },
    )

    document_chain = create_stuff_documents_chain(model, prompt)
    chain = create_retrieval_chain(retriever, document_chain)
    
    #
    return chain


def ask(query: str, model: str, vector_store: Chroma):
    chain = rag_chain(model=model, vector_store=vector_store)
    # invoke chain
    result = chain.invoke({"input": query})
    return result



async def process_and_store_file(
    files: List[UploadFile],
    dataset_name: str = "rag_dataset",
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
    user_id: str = None
) -> dict:
    try:
        splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    except Exception as e:
        raise ValueError(f"Error initializing text splitter: {e}")
    
    all_results = []
    all_documents = []

    for f in files:
        # 1) Read bytes and wrap in BytesIO

        try:
            data = await f.read()
        except Exception as e:
            raise ValueError(f"Error reading file {f.filename}: {e}")

        # 2) Partition into elements

        try:
            elems = partition(file=BytesIO(data), extract_images_in_pdf=True, encoding='utf-8')
        except Exception as e:
            raise ValueError(f"Error partitioning file {f.filename}: {e}")

        text = ""
        try:
            text_segments = [
                el.text
                for el in elems
                if getattr(el, 'text', None)   # attribute exists
                and isinstance(el.text, str)    # it really is a string
                and el.text.strip()             # non‐empty after stripping
            ]
            text = "\n".join(text_segments)


            # 4) Split into manageable chunks
            chunks = splitter.split_text(text)
        except Exception as e:
            raise ValueError(f"Error processing text from file {f.filename}: {e}")
        
        # 5) Convert chunks to Document objects

        all_documents.extend(
            [Document(page_content=chunk, metadata={"source": f.filename})
            for chunk in chunks
        ])

        try:
            # 6) Embed all chunks in one batch
            embeddings = ST_MODEL.encode(
                chunks,
                convert_to_tensor=True,
                normalize_embeddings=True,
            )
            embeddings = embeddings.cpu().tolist()  # JSON‑serialize
        except Exception as e:
            raise ValueError(f"Error embedding chunks from file {f.filename}: {e}")

        # 7) Collect results

        all_results.append({
            'file': f.filename, 
            'results': [{'text': c, 'embedding': e} for c, e in zip(chunks, embeddings)]
        })

    
    # 8) Store all results in the database
    try:
        db.datasets_rag.insert_one({
            "owner": user_id,
            "name": dataset_name,
            "chunks": all_results,
            "createdAt": datetime.now(pytz.utc),
        })
    except Exception as e:
        raise ValueError(f"Error storing dataset in database: {e}")

    return {
        'status': 'ok'
    }



def process_files_job(
    files_data: List[UploadFile],
    dataset_name: str,
    chunk_size: int,
    chunk_overlap: int,
    user_id: str,
):
    """
    Synchronous function called by RQ worker.
    - files_data: list of raw bytes
    - files_meta: list of {filename: str}
    """
    # Reconstruct UploadFile-like objects if needed, or change
    # your process_and_store to accept raw bytes+meta instead of UploadFile.
    for file in files_data:
        # call your existing logic file‐by‐file
        process_and_store_file(
            data_bytes=file,
            dataset_name=dataset_name,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            user_id=user_id,
        )
    return {"status": "completed"}

    
async def run_query(
        dataset_id: str,
        query: str,
        model: str = "mistral"
):
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


    # Create Chroma vector store with Document objects
    vector_store = Chroma(
        embedding_function=SentenceTransformerEmbeddings(ST_MODEL)
    )

    vector_store.add_texts(
        texts = texts,
        metadatas = metadatas,
        embeddings = embeddings
    )

    return ask(query=query, model=model, vector_store=vector_store)