from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
import os
from dotenv import load_dotenv

def process_pdfs():
    # Initialize embeddings with explicit model name to avoid deprecation warning
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
    
    # Initialize an empty list to store all documents
    all_documents = []
    
    # Process each PDF in the data directory
    pdf_dir = "data"  # Adjust this path as needed
    for filename in os.listdir(pdf_dir):
        if filename.endswith(".pdf"):
            pdf_path = os.path.join(pdf_dir, filename)
            print(f"Processing {filename}...")
            
            # Load PDF
            loader = PyPDFLoader(pdf_path)
            documents = loader.load()
            
            # Split documents
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200
            )
            split_docs = text_splitter.split_documents(documents)
            all_documents.extend(split_docs)
    
    # Create and save FAISS index
    print("Creating FAISS index...")
    vectorstore = FAISS.from_documents(all_documents, embeddings)
    vectorstore.save_local("faiss_index")
    print("Index saved successfully!")

if __name__ == "__main__":
    load_dotenv()
    process_pdfs()