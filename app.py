import streamlit as st
import shutil
import warnings
warnings.filterwarnings("ignore")

from processing.extractionfrompdf import extract_all_pdfs
from processing.chunking import chunk_texts
from processing.embedding import create_vectorstore
from rag.ragchain import RAGChain
from transformers import logging
logging.set_verbosity_error()

MODEL_PATH = "/home/becker21/llmmodels/qwen3-8b-q4_k_m.gguf"
VECTORSTORE_PATH = "rag_vectorstore"
DATA_PATH = "data"
DIMENSION = "Plausibility"

st.set_page_config(page_title="Interactive Data Quality Questionnaire For AI In Medicine", layout="centered")
st.title("Interactive Data Quality Questionnaire For AI In Medicine")
st.caption(f"Dimension: {DIMENSION}")

@st.cache_resource
def load_rag_chain():
    texts = extract_all_pdfs(DATA_PATH)
    chunks = chunk_texts(texts)
    #shutil.rmtree(VECTORSTORE_PATH, ignore_errors=True)
    create_vectorstore(chunks)
    rag_chain = RAGChain(
        vectorstore_path=VECTORSTORE_PATH,
        llama_model_path=MODEL_PATH
    )
    return rag_chain

if "rag_chain" not in st.session_state:
    with st.spinner("Loading RAG system and model... this may take a minute."):
        st.session_state.rag_chain = load_rag_chain()

if "history" not in st.session_state:
    st.session_state.history = []

if "current_question" not in st.session_state:
    st.session_state.current_question = None

if "current_source" not in st.session_state:
    st.session_state.current_source = None

if "generated_questions" not in st.session_state:
    st.session_state.generated_questions = []

if "dataset_info_list" not in st.session_state:
    st.session_state.dataset_info_list = []

if "used_chunks" not in st.session_state:
    st.session_state.used_chunks = set()

if "questionnaire_finished" not in st.session_state:
    st.session_state.questionnaire_finished = False

def generate_next_question(answer_text):
    st.session_state.dataset_info_list.append(answer_text)
    combined = "\n".join(st.session_state.dataset_info_list)

    search_query = f"Dataset description: {combined} Focus: {DIMENSION} Already covered: {' '.join(st.session_state.generated_questions)}"

    retrieved_chunks = st.session_state.rag_chain.retrieve_chunks(search_query, top_k=10)
    retrieved_chunks = [c for c in retrieved_chunks if c.page_content not in st.session_state.used_chunks]
    for c in retrieved_chunks:
        st.session_state.used_chunks.add(c.page_content)

    if not retrieved_chunks:
        return None, None

    context_text = "\n".join([
        f"{chunk.page_content.strip()} (Source: {chunk.metadata.get('source', 'unknown')})"
        for chunk in retrieved_chunks
    ])

    raw_output = st.session_state.rag_chain.generate_question(
        context_chunks=context_text,
        dataset_info=combined,
        generated_questions=st.session_state.generated_questions,
        dimension=DIMENSION
    )

    question = ""
    source = ""
    for line in raw_output.splitlines():
        if "Next question" in line:
            question = line.replace("Next question :", "").replace("Next question:", "").strip()
        if "Source:" in line:
            source = line.replace("Source:", "").strip()

    if question:
        st.session_state.generated_questions.append(question)

    return question, source

# Show history
if st.session_state.history:
    st.subheader("Previous questions")
    for item in st.session_state.history:
        with st.container():
            st.markdown(f"**Q:** {item['question']}")
            st.markdown(f"**A:** {item['answer']}")
            if item.get('source'):
                st.caption(f"Source: {item['source']}")
            st.divider()

# First input or current question
if st.session_state.current_question is None and not st.session_state.history:
    st.subheader("Which data types exist in your data set?")
    dataset_description = st.text_area("Describe your dataset in a few words:", key="initial_input")
    if st.button("Start questionnaire"):
        if dataset_description.strip():
            with st.spinner("Generating first question..."):
                question, source = generate_next_question(dataset_description)
            if question:
                st.session_state.current_question = question
                st.session_state.current_source = source
                st.session_state.history.append({
                    "question": "Which data types exist in your data set?",
                    "answer": dataset_description,
                    "source": None
                })
                st.rerun()

elif st.session_state.questionnaire_finished:

    st.success("Questionnaire finished.")

    st.subheader("Questionnaire Summary")

    summary_text = ""

    for item in st.session_state.history:
        st.markdown(f"**Q:** {item['question']}")
        st.markdown(f"**A:** {item['answer']}")

        summary_text += f"Q: {item['question']}\n"
        summary_text += f"A: {item['answer']}\n\n"

        if item.get("source"):
            st.caption(f"Source: {item['source']}")
            summary_text += f"Source: {item['source']}\n\n"

        st.divider()

    st.download_button(
        label="Download questionnaire",
        data=summary_text,
        file_name="questionnaire_summary.txt",
        mime="text/plain"
    )

    if st.button("Start new questionnaire"):
        for key in list(st.session_state.keys()):
            del st.session_state[key]

        st.rerun()                

elif st.session_state.current_question:
    st.subheader("Current question")
    st.markdown(f"**{st.session_state.current_question}**")
    if st.session_state.current_source:
        st.caption(f"Source: {st.session_state.current_source}")

    col1, col2, col3, col4 = st.columns(4)
    answer_button = None
    with col1:
        if st.button("Yes", use_container_width=True):
            answer_button = "Yes"
    with col2:
        if st.button("No", use_container_width=True):
            answer_button = "No"
    with col3:
        if st.button("Don't know", use_container_width=True):
            answer_button = "Don't know"
    with col4:
        if st.button("Exit", use_container_width=True):
            st.session_state.questionnaire_finished = True
            st.rerun()

    extra = st.text_area("Add more details to your answer (optional)", key="extra_input")

    if answer_button:
        full_answer = answer_button
        if extra.strip():
            full_answer += f" — {extra.strip()}"

        st.session_state.history.append({
            "question": st.session_state.current_question,
            "answer": full_answer,
            "source": st.session_state.current_source
        })

        with st.spinner("Generating next question..."):
            question, source = generate_next_question(full_answer)

        if question:
            st.session_state.current_question = question
            st.session_state.current_source = source
        else:
            st.session_state.current_question = None
            st.success("Questionnaire complete!")

        st.rerun()