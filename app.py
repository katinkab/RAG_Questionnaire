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

MODEL_PATH = "/scratch2/fast/becker21/models/Qwen_Qwen3-32B-Q4_K_M.gguf"
VECTORSTORE_PATH = "rag_vectorstore"
DATA_PATH = "data"
DIMENSIONS = [
    ("Plausibility", "The extent to which data values match real world knowledge."),
    ("Traceability", "The extent to which data origins and transformations are documented, data lineage is available."),
    ("Compliance", "The extent to which the dataset and data values are in accordance with laws, regulations or standards.")
]

st.set_page_config(page_title="Interactive Data Quality Questionnaire For AI In Medicine", layout="centered")
st.markdown("""
    <style>
    .block-container {
        max-width: 900px;
        padding-left: 2rem;
        padding-right: 2rem;
    }
    </style>
""", unsafe_allow_html=True)
st.title("Interactive Data Quality Questionnaire For AI In Medicine")
#st.caption(f"Dimension: {DIMENSION}")


@st.cache_resource
def load_rag_chain():
    texts = extract_all_pdfs(DATA_PATH)
    chunks = chunk_texts(texts)
    shutil.rmtree(VECTORSTORE_PATH, ignore_errors=True)
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

#if "depth" not in st.session_state:
#    st.session_state.depth = 0

#if "current_context" not in st.session_state:
#    st.session_state.current_context = None

#if "satisfaction" not in st.session_state:
#    st.session_state.satisfaction = []

if "input_counter" not in st.session_state:
    st.session_state.input_counter = 0

if "dimension_index" not in st.session_state:
    st.session_state.dimension_index = 0



def current_dimension():
    return DIMENSIONS[st.session_state.dimension_index][0]

def current_dimension_def():
    return DIMENSIONS[st.session_state.dimension_index][1]

def add_dimension_to_history():
    st.session_state.history.append({
        "type": "dimension",
        "dimension": current_dimension(),
        "definition": current_dimension_def()
    })

def generate_next_question(answer_text):
    st.session_state.dataset_info_list.append(answer_text)
    combined = "\n".join(st.session_state.dataset_info_list)

    if st.session_state.generated_questions:
        expanded = st.session_state.rag_chain.expand_answer(
            answer=answer_text,
            last_question=st.session_state.generated_questions[-1],
            dataset_info=combined
        )
    else:
        expanded = answer_text

    already_covered = st.session_state.generated_questions
    search_query = f"Dataset description: {combined} Expanded context: {expanded} Focus: {current_dimension()} Already covered (do not retrieve similar content): {' '.join(already_covered)}"

    retrieved_chunks = st.session_state.rag_chain.retrieve_chunks(search_query, top_k=3)
    retrieved_chunks = [c for c in retrieved_chunks if c.page_content not in st.session_state.used_chunks]
    for c in retrieved_chunks:
        st.session_state.used_chunks.add(c.page_content)

    if not retrieved_chunks:
        return None, None

    context_text = "\n".join([
        f"{chunk.page_content.strip()} (Source: {chunk.metadata.get('source', 'unknown')})"
        for chunk in retrieved_chunks
    ])
    st.session_state.current_context = context_text  
    st.session_state.last_context = context_text


    raw_output = st.session_state.rag_chain.generate_question(
        context_chunks=context_text,
        dataset_info=combined,
        generated_questions=st.session_state.generated_questions,
        dimension=current_dimension()
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

def parse_question(raw_output):
    question = ""
    source = ""
    for line in raw_output.splitlines():
        if "Next question" in line:
            question = line.replace("Next question :", "").replace("Next question:", "").strip()
        if "Source:" in line:
            source = line.replace("Source:", "").strip()
    return question, source

# Show history
for item in st.session_state.history:
        with st.container():

            #Dimension header block
            if item.get("type") == "dimension":
                st.subheader(f"Dimension: {item['dimension']}")
                st.caption(item["definition"])
                st.divider()
                continue

            st.markdown(f"**Q:** {item['question']}")
            st.markdown(f"**A:** {item['answer']}")
            #if item.get('satisfactory') is not None:
            #    indicator = "🟢" if item['satisfactory'] else "🔴"
            #    if "Don't know" in item['answer']:
            #        indicator = "🟡"
            #    st.markdown(f"**A:** {indicator} {item['answer']}")
            #else:
            #    st.markdown(f"**A:** {item['answer']}")
            if item.get('source'):
                st.caption(f"Source: {item['source']}")
            st.markdown("<hr style='margin: 4px 0; border-color: #eee;'>", unsafe_allow_html=True)

# First input or current question
if st.session_state.current_question is None and not st.session_state.history:
    st.subheader("Which data types exist in your data set?")
    dataset_description = st.text_area("Describe your dataset in a few words:", key="initial_input")
    if st.button("Start questionnaire"):
        if dataset_description.strip():

            st.session_state.dimension_index = 0

            # ✅ ADD THIS HERE (first dimension header)
            st.session_state.history.append({
                "type": "dimension",
                "dimension": current_dimension(),
                "definition": current_dimension_def()
            })

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
        #Dimension header block
        if item.get("type") == "dimension":
            st.subheader(f"Dimension: {item['dimension']}")
            st.caption(item["definition"])
            st.divider()
            continue

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

    st.caption(
        f"Dimension: {current_dimension()} — {current_dimension_def()}"
    )

    st.subheader("Current question")
    st.markdown(f"**{st.session_state.current_question}**")
    if st.session_state.current_source:
        st.caption(f"Source: {st.session_state.current_source}")
    if st.session_state.get('last_context'):
        with st.expander("Show retrieved context"):
            st.text(st.session_state.last_context)

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

    extra = st.text_area("Add more details to your answer (optional)", key=f"extra_input_{st.session_state.input_counter}")

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
            next_index = st.session_state.dimension_index + 1

            if next_index < len(DIMENSIONS):
                st.session_state.dimension_index = next_index

                # reset retrieval state for the new dimension
                st.session_state.generated_questions = []
                st.session_state.used_chunks = set()

                st.session_state.history.append({
                    "type": "dimension",
                    "dimension": current_dimension(),
                    "definition": current_dimension_def()
                })

                # immediately start the next dimension
                question, source = generate_next_question(full_answer)

                if question:
                    st.session_state.current_question = question
                    st.session_state.current_source = source
                else:
                    st.session_state.questionnaire_finished = True

            else:
                st.session_state.questionnaire_finished = True

        st.session_state.input_counter += 1
        st.rerun()