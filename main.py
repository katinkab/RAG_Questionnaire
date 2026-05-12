# main.py
import warnings
warnings.filterwarnings("ignore")
import shutil
from processing.extractionfrompdf import extract_all_pdfs
from processing.chunking import chunk_texts
from processing.embedding import create_vectorstore
from rag.ragchain import RAGChain
from transformers import logging
logging.set_verbosity_error()  # hide warnings from Huggingface


#-----------------------------
# PDF extraction
#-----------------------------
print("Starting PDF extraction...")
texts = extract_all_pdfs("data")
print(f"Extracted text from {len(texts)} PDFs.\nDone.\n")

#-----------------------------
# Chunking
#-----------------------------
print("Chunking text...")
chunks = chunk_texts(texts)
print(f"Created {len(chunks)} chunks.\nDone.\n")

#-----------------------------
# Vectorstore creation
#-----------------------------
print("Creating vectorstore...")
shutil.rmtree("rag_vectorstore", ignore_errors=True)  #delete old vectorstore if exists
vectorstore = create_vectorstore(chunks)
print("Vectorstore ready.\n")

#-----------------------------
# Setup RAG chain
#-----------------------------
print("Setting up RAG chain...")
rag_chain = RAGChain(
    vectorstore_path="rag_vectorstore",
    llama_model_path="/scratch2/fast/becker21/models/Qwen_Qwen3-32B-Q4_K_M.gguf"
)
print("RAG chain ready.\n")

#-----------------------------
# Interactive questionnaire generation
#-----------------------------
print("--- Interactive Questionnaire Generation ---")
print("\nWelcome to the interactive questionnaire for qualitative data quality assessment. Please provide information about your dataset by answering the questions provided.")
print("\nWhich data types exist in your data?\n")
all_dataset_info = []
generated_questions = []
dimension = "Plausibility"  #choose different dimension here if needed

while True:
    dataset_info = input("Enter dataset info / answer to previous question (or 'exit' to finish): ")
    if dataset_info.lower() == "exit":
        print("\nQuestionnaire generation finished.")
        break

    all_dataset_info.append(dataset_info)
    combined_dataset_info = "\n".join(all_dataset_info)

        # expand the answer before retrieval
    if generated_questions:  # only expand if there was a previous question
        expanded = rag_chain.expand_answer(
            answer=dataset_info,
            last_question=generated_questions[-1],
            dataset_info=combined_dataset_info
        )
    else:
        expanded = dataset_info

    # then use expanded instead of combined_dataset_info for the search query
    search_query = f"""
    Dataset description:
    {expanded}

    Focus: {dimension} (Definition: The extent to which data values match real world knowledge.)
    """

    #Retrieve top k relevant chunks from vectorstore
    #Right now only based on user input, this gets difficult if this is only answer yes/no
    retrieved_chunks = rag_chain.retrieve_chunks(search_query, top_k=3)
    if not retrieved_chunks:
        print("No relevant chunks found for this input.\n")
        continue

    context_text = "\n".join([f"{chunk.page_content.strip()} (Source: {chunk.metadata.get('source', 'unknown')})" for chunk in retrieved_chunks])
    print("**********Chunks*****************************",)
    print("*****This is the context found:**************\n", context_text, "\n")
    print("*********************************************")

    #Generate next question using the interactive prompt
    next_question = rag_chain.generate_question(
        context_chunks=context_text,
        dataset_info=combined_dataset_info,
        generated_questions=generated_questions,
        dimension=dimension
    )

    print("---------------------------------------\n", next_question, "\n")

    question_text = next_question.replace("Next question :", "").strip()
    generated_questions.append(question_text)
