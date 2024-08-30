import os
import time
import tempfile
import torch
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory

## Contextual Recompression
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import DocumentCompressorPipeline
from langchain_community.document_transformers import EmbeddingsRedundantFilter
from langchain.retrievers.document_compressors import EmbeddingsFilter

#Long Context Reorder
from LCR_Retriever import LongContextReorderRetriever

# Function to load PDFs from a folder and count the number of files loaded
def load_pdfs_from_folder(folder_path):
    pdf_documents = []
    loaded_files = 0
    for file_name in os.listdir(folder_path):
        if file_name.endswith(".pdf"):
            file_path = os.path.join(folder_path, file_name)
            pdf_documents.extend(PyPDFLoader(file_path).load())
            loaded_files += 1
    return pdf_documents, loaded_files

# Function to embed PDF documents from a folder into a vectorDB ONLY FOR LOCAL USE
def build_vector_db_local(folder):
    # Step 1: Read documents
    docs = []
    start_time = time.time()
    pdfs, loaded_files = load_pdfs_from_folder(folder)
    docs = pdfs
    step1_time = time.time() - start_time
    print(f"Loading pdfs from folder time: {step1_time:.2f} seconds")
    print(f"Loaded {loaded_files} pdfs")
    
    # Step 2: Split documents
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=200)
    start_time = time.time()
    splits = text_splitter.split_documents(docs)
    step2_time = time.time() - start_time
    
    print(f"Splitting documents time {step2_time:.2f} seconds")
    print(f"Length of splits {(len(splits))}")
    print(f"Length of docs {(len(docs))}")
    
    # Step 3: Create embeddings form docs and store them in a vector DB
    model_kwargs = {'device': 'cuda:0' if torch.cuda.is_available() else 'cpu'}
    embeddings = HuggingFaceEmbeddings(model_name="intfloat/e5-large-v2", model_kwargs = model_kwargs)
    start_time = time.time()
    vectordb = FAISS.from_documents(splits, embeddings)
    step3_time = time.time() - start_time
    print(f"Loading embeddings in vector DB took {step3_time:.2f} seconds")
    
    return vectordb

# Function to embed PDF documents in a VectorDB FOR STREAMLIT ONLINE APP
def configure_vector_db(uploaded_files):
    # Step 1: Read documents
    docs = []
    temp_dir = tempfile.TemporaryDirectory()
    
    for file in uploaded_files:
        temp_filepath = os.path.join(temp_dir.name, file.name)
        with open(temp_filepath, "wb") as f:
            f.write(file.getvalue())
        docs.extend(PyPDFLoader(temp_filepath).load())
    
    # Print the number of documents loaded
    loaded_files = len(uploaded_files)
    print(f"Loaded {loaded_files} pdfs")

    # Step 2: Split documents
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=200)
    start_time = time.time()
    splits = text_splitter.split_documents(docs)
    step2_time = time.time() - start_time
    
    print(f"Splitting documents time {step2_time:.2f} seconds")
    print(f"Length of splits {len(splits)}")
    print(f"Length of docs {len(docs)}")
    
    # Step 3: Create embeddings and store in vector DB
    model_kwargs = {'device': 'cuda:0' if torch.cuda.is_available() else 'cpu'}
    embeddings = HuggingFaceEmbeddings(model_name="intfloat/e5-large-v2", model_kwargs=model_kwargs)
    start_time = time.time()
    vectordb = FAISS.from_documents(splits, embeddings)
    step3_time = time.time() - start_time
    print(f"Loading embeddings in vector DB took {step3_time:.2f} seconds")
    
    return vectordb

## Function to build the retriever given the vectord DB, number of retrieved chunks k and retreival method
def build_retriever(vectordb, k , retrieval):    
    # Define retriever
    base_retriever = vectordb.as_retriever(search_kwargs={"k": k})
    model_kwargs = {'device': 'cuda:0' if torch.cuda.is_available() else 'cpu'}
    embeddings = HuggingFaceEmbeddings(model_name="intfloat/e5-large-v2", model_kwargs = model_kwargs)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=80)
    
    if retrieval == "Classic":
        retriever = base_retriever       
    elif retrieval == "Compression":
        redundant_filter = EmbeddingsRedundantFilter(embeddings=embeddings)
        relevant_filter = EmbeddingsFilter(embeddings=embeddings, similarity_threshold=0.76)
        pipeline_compressor = DocumentCompressorPipeline(
        transformers=[text_splitter, redundant_filter, relevant_filter]
        )
        retriever = ContextualCompressionRetriever(
        base_compressor=pipeline_compressor, base_retriever=base_retriever
        )        
    elif retrieval == "Reorder":
        retriever = LongContextReorderRetriever(base_retriever = base_retriever)   
    else:
        import warnings
        warnings.warn("You must select one between Classic, Compression, Reorder ", UserWarning)
        raise ValueError("Invalid retriever selected. Please choose between Classic, Compression, Reorder.")
        
    return retriever        
    
