# RAG Questionnaire

Generate questions in an interactive questionnaire
Use RAG (retriever + LLM) to generate a customized question.

**Pipeline**

PDF text extraction:
Read text from scientific PDFs.

Chunk & clean text:
Split long text into chunks suitable for embedding.

Create embeddings:
Encode chunks into vectors.
Build a vector store for fast retrieval.

Retrieve relevant chunks:
Based on a user prompt
