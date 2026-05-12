# processing/embedding.py
# saves chunks in FAISS vectorstore
# optional improvement: change sentence-transformer from lightweight option to more specific medical?

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

def create_vectorstore(chunks, save_path="rag_vectorstore"):
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = FAISS.from_documents(chunks, embeddings)
    vectorstore.save_local(save_path)
    return vectorstore

def load_vectorstore(path="rag_vectorstore"):
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return FAISS.load_local(path, embeddings, allow_dangerous_deserialization=True)