# processing/chunking.py
from langchain_text_splitters import RecursiveCharacterTextSplitter

# use smaller chunks so model does not get confused
# use overlap for orientation
def chunk_texts(text_dict, chunk_size=350, chunk_overlap=80):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )

    chunks = []
    for filename, text in text_dict.items():
        docs = splitter.create_documents([text], metadatas=[{"source": filename}])
        chunks.extend(docs)
    return chunks
