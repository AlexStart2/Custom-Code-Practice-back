from unstructured.partition.auto import partition
import torch
from sentence_transformers import SentenceTransformer
from langchain.text_splitter import CharacterTextSplitter
from langchain_core.embeddings import Embeddings
from io import BytesIO
from langchain_community.chat_models import ChatOllama
from langchain_core.prompts import PromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.vectorstores import Chroma
from langchain.chains import create_retrieval_chain
from langchain_core.documents import Document       
from typing import List
from fastapi import UploadFile
from db import db
from datetime import datetime
import pytz
from bson.objectid import ObjectId


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
    
EMBEDDING_MODEL = SentenceTransformerEmbeddings(ST_MODEL)


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
    #
    chain = rag_chain(model=model, vector_store=vector_store)
    # invoke chain
    result = chain.invoke({"input": query})
    return result



async def process_and_store(
    files: List[UploadFile],
    dataset_name: str = "rag_dataset",
    user_id: str = None
) -> dict:
    splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    all_results = []
    all_documents = []

    for f in files:
        # 1) Read bytes and wrap in BytesIO
        data = await f.read()
        # 2) Partition into elements
        elems = partition(file=BytesIO(data))

        # 3) Extract text from all elements
        text = "\n".join(el.text for el in elems if hasattr(el, 'text'))

        # 4) Split into manageable chunks
        chunks = splitter.split_text(text)
        
        # 5) Convert chunks to Document objects

        all_documents.extend(
            [Document(page_content=chunk, metadata={"source": f.filename})
            for chunk in chunks
        ])

        # 6) Embed all chunks in one batch
        embeddings = ST_MODEL.encode(
            chunks,
            convert_to_tensor=True,
            normalize_embeddings=True
        )
        embeddings = embeddings.cpu().tolist()  # JSON‑serialize

        # 7) Collect results

        all_results.append({
            'file': f.filename, 
            'results': [{'text': c, 'embedding': e} for c, e in zip(chunks, embeddings)]
        })

    
    # 8) Store all results in the database
    db.datasets_rag.insert_one({
        "owner": user_id,
        "name": dataset_name,
        "chunks": all_results,
        "createdAt": datetime.now(pytz.utc),
    })


    return {
        'status': 'ok'
    }

    
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
        embedding_function=EMBEDDING_MODEL,
        # persist_directory="./"+dataset_id+"_chroma_db"
    )

    vector_store.add_texts(
        texts = texts,
        metadatas = metadatas,
        embeddings = embeddings
    )

    return ask(query=query, model=model, vector_store=vector_store)