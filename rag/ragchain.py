import os
from processing.retriever import get_retriever
from langchain_community.llms import LlamaCpp
from langchain_classic.chains import LLMChain
from prompts.prompts import QUESTIONNAIRE_PROMPT


class RAGChain:
    def __init__(self, vectorstore_path: str, llama_model_path: str, k: int = 2):
        # --- Load retriever ---
        if not os.path.exists(vectorstore_path):
            raise ValueError(f"Vectorstore path not found: {vectorstore_path}")
        self.retriever = get_retriever(vectorstore_path, k=k)

        # --- Load model ---
        if not os.path.exists(llama_model_path):
            raise ValueError(f"Model path not found: {llama_model_path}")
        self.llm = LlamaCpp(
            model_path=llama_model_path,
            n_ctx=1024,
            temperature=0.2,
            max_tokens=50,
            n_threads=16,
            n_batch=1024,
            n_gpu_layers=-1,
	    repeat_penalty=1.2,
            verbose=False
        )

        # --- Load LLMChain with interactive prompt ---
        self.qa_chain = LLMChain(
            llm=self.llm,
            prompt=QUESTIONNAIRE_PROMPT,
            verbose=False
        )

    # -----------------------------
    # Retrieve top-k relevant chunks (MMR, not similarity search)
    # -----------------------------
    def retrieve_chunks(self, dataset_info: str, top_k: int = None):
        if top_k is None:
            top_k = self.retriever.search_kwargs.get("k", 2)
        self.retriever.search_kwargs["k"] = top_k
        return self.retriever.invoke(dataset_info)
    # -----------------------------
    # Generate a single question
    # -----------------------------
    def generate_question(self, context_chunks: str, dataset_info: str, generated_questions: list, dimension: str):
        prompt_text = QUESTIONNAIRE_PROMPT.format(
            context_chunks=context_chunks,
            dataset_info=dataset_info,
            generated_questions="\n".join(generated_questions),
            dimension=dimension
        )
    
        llm_result = self.llm.generate([prompt_text])
    
        return llm_result.generations[0][0].text

    def expand_answer(self, answer: str, last_question: str, dataset_info: str) -> str:
        prompt = f"""<|im_start|>system
        You are a data quality expert. Expand a short answer into a detailed description.<|im_end|>
        <|im_start|>user
        The user was asked: {last_question}
        The user answered: {answer}
        Dataset context: {dataset_info}

        Write 2-3 sentences expanding this answer with relevant data quality context.
        <|im_end|>
        <|im_start|>assistant
        <think>

        </think>

        """
        result = self.llm.generate([prompt])
        return result.generations[0][0].text
