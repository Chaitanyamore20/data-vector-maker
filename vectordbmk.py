from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain_community.embeddings.spacy_embeddings import SpacyEmbeddings
import spacy
import os
import logging

PROCESSED_DATA_DIR = "processed_data"

# Load the spaCy model
embeddings = SpacyEmbeddings(model_name="en_core_web_sm")
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# Step 1: Split Text into Chunks
def load_and_split_text():
    documents = []
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=512,
        chunk_overlap=50  
    )

    for file in os.listdir(PROCESSED_DATA_DIR):
        file_path = os.path.join(PROCESSED_DATA_DIR, file)
        with open(file_path, 'r', encoding='utf-8') as f:
            raw_text = f.read()
        
        chunks = text_splitter.split_text(raw_text)
        for chunk in chunks:
            documents.append(Document(page_content=chunk, metadata={"source": file}))

    logging.info(f"Loaded and split text into {len(documents)} chunks.")
    return documents

# Step 2: Embed and Store Data in FAISS
def embed_and_store_in_faiss(documents):
    faiss_store = FAISS.from_documents(documents, embedding=embeddings)
    faiss_store.save_local("faiss_index")
    logging.info("Data embedded and stored in FAISS vector store.")

# Step 3: Load FAISS for Retrieval and Load FAISS store from local storage
def load_faiss_store():
    return FAISS.load_local("faiss_index", embeddings)

# Step 4: Query FAISS for RAG and Retrieve top-k relevant documents
def query_faiss_store(query, faiss_store, top_k=3):
    docs = faiss_store.similarity_search(query, k=top_k)
    for i, doc in enumerate(docs):
        logging.info(f"Result {i+1}: {doc.page_content}\nSource: {doc.metadata['source']}")
    return docs

logging.info("Starting text preprocessing and embedding...")
documents = load_and_split_text()
embed_and_store_in_faiss(documents)


