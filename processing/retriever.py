# processing/retriever.py
from processing.embedding import load_vectorstore

def get_retriever(vectorstore_path="rag_vectorstore", k=2):
    #Load vectorstore and return retriever with mmr to explore vectorspace
    vectorstore = load_vectorstore(vectorstore_path)
    retriever = vectorstore.as_retriever(
        #before I tried search_type="similarity", fine but got stuck in same topic
        search_type="mmr",
        search_kwargs={
		"k": k,
		"fetch_k": 40,
		"lambda_mult": 0.6
	}
    )
    return retriever
