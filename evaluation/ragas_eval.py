"""
evaluation/ragas_eval.py

RAGAS evaluation 
  - no faithfulness (claims supported by context makes no sense since no Q&A) 
  - context_precision (reference-free) (chunks useful for question generation?) 
  - AspectCritic (chunk relevant for context)

FAISS vectorstore must already exist at VECTORSTORE_PATH

Install:
    pip install "ragas>=0.2,<0.3" langchain-huggingface datasets pandas
"""

import os
import sys
import time

# Make this runnable both as `python evaluation/ragas_eval.py` (script's own
# folder ends up on sys.path, not the project root) and as
# `python -m evaluation.ragas_eval` (project root ends up on sys.path).
# Without this, `from rag.ragchain import RAGChain` below fails with
# ModuleNotFoundError when run the first way.
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from datasets import Dataset

from langchain_community.llms import LlamaCpp
from langchain_huggingface import HuggingFaceEmbeddings

from ragas import evaluate, RunConfig
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.metrics import LLMContextPrecisionWithoutReference, AspectCritic

from rag.ragchain import RAGChain

# ---------------------------------------------------------------------------
# Config -- mirror app.py's constants. If you refactor these into a shared
# config module later, import from there instead of redefining here.
# ---------------------------------------------------------------------------
MODEL_PATH = "/scratch2/fast/becker21/models/Qwen_Qwen3-32B-Q4_K_M.gguf"
VECTORSTORE_PATH = "rag_vectorstore"
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"  # must match processing/embedding.py

DIMENSIONS = [
    ("Plausibility", "The extent to which data values match real world knowledge."),
    ("Traceability", "The extent to which data origins and transformations are documented, data lineage is available."),
    ("Compliance", "The extent to which the dataset and data values are in accordance with laws, regulations or standards.")
]

OUT_CSV = os.path.join(os.path.dirname(__file__), "ragas_results.csv")


# ---------------------------------------------------------------------------
# 1. Judge LLM -- a SEPARATE LlamaCpp instance, bigger context/output budget.
#    See the module docstring for why this can't just be rag_chain.llm.
# ---------------------------------------------------------------------------
def build_judge_llm(model_path: str = MODEL_PATH):
    judge_llm = LlamaCpp(
        model_path=model_path,
        n_ctx=8192,          # room for retrieved context + judge prompt + reasoning
        temperature=0.0,     # judging should be deterministic
        max_tokens=1000,      # enough room for verdict + brief reasoning, unlike the 50-token generation config
        n_threads=16,
        n_batch=2048,
        n_gpu_layers=-1,
        repeat_penalty=1.2,
        verbose=False
    )
    return LangchainLLMWrapper(judge_llm)


# ---------------------------------------------------------------------------
# 2. Judge embeddings -- same model as the vectorstore, for consistency.
# ---------------------------------------------------------------------------
def build_judge_embeddings(model_name: str = EMBEDDING_MODEL_NAME):
    hf_embeddings = HuggingFaceEmbeddings(model_name=model_name)
    return LangchainEmbeddingsWrapper(hf_embeddings)


# ---------------------------------------------------------------------------
# 3. Helpers duplicated from app.py's generate_next_question / parse_question.
#    Kept in sync manually for now -- if app.py's search_query construction
#    or output parsing changes, update here too. (Consider factoring both
#    into a shared module, e.g. rag/query_utils.py, imported by both
#    app.py and this script, so they can't drift apart.)
# ---------------------------------------------------------------------------
def build_search_query(dataset_info: str, dimension: str, expanded: str, already_covered: list[str]) -> str:
    return (
        f"Dataset description: {dataset_info} "
        f"Expanded context: {expanded} "
        f"Focus: {dimension} "
        f"Already covered (do not retrieve similar content): {' '.join(already_covered)}"
    )


def parse_question(raw_output: str) -> tuple[str, str]:
    question, source = "", ""
    for line in raw_output.splitlines():
        if "Next question" in line:
            question = line.replace("Next question :", "").replace("Next question:", "").strip()
        if "Source:" in line:
            source = line.replace("Source:", "").strip()
    return question, source


# ---------------------------------------------------------------------------
# 4. Collect eval samples by driving RAGChain directly (no Streamlit needed).
#
#    scenarios: list of dicts, e.g.
#      {"dataset_info": "De-identified ICU vitals dataset from 3 hospitals",
#       "dimension": "Plausibility"}
#
#    For v1 this covers the FIRST question generated per dimension (empty
#    generated_questions history, so expand_answer is skipped, matching
#    app.py's behavior where `expanded = answer_text` on the first turn).
#    Extend with a "history" field per scenario later to cover follow-up
#    turns -- the loop below is written so that's a small addition.
# ---------------------------------------------------------------------------
def collect_eval_samples(rag_chain: RAGChain, scenarios: list[dict]) -> Dataset:
    rows = {"user_input": [], "response": [], "retrieved_contexts": []}

    for scenario in scenarios:
        t0 = time.time()
        dataset_info = scenario["dataset_info"]
        dimension = scenario["dimension"]
        generated_questions = scenario.get("generated_questions", [])
        already_covered = generated_questions

        expanded = dataset_info if not generated_questions else rag_chain.expand_answer(
            answer=dataset_info,
            last_question=generated_questions[-1],
            dataset_info=dataset_info,
        )

        search_query = build_search_query(dataset_info, dimension, expanded, already_covered)

        print(f"  [{dimension}] retrieving chunks...", flush=True)
        retrieved_chunks = rag_chain.retrieve_chunks(search_query, top_k=3)
        if not retrieved_chunks:
            print(f"[skip] no chunks retrieved for dimension={dimension!r}, dataset_info={dataset_info!r}")
            continue
        print(f"  [{dimension}] got {len(retrieved_chunks)} chunks in {time.time()-t0:.1f}s, generating question...", flush=True)

        context_text = "\n".join([
            f"{chunk.page_content.strip()} (Source: {chunk.metadata.get('source', 'unknown')})"
            for chunk in retrieved_chunks
        ])

        t1 = time.time()
        raw_output = rag_chain.generate_question(
            context_chunks=context_text,
            dataset_info=dataset_info,
            generated_questions=generated_questions,
            dimension=dimension,
        )
        question, _source = parse_question(raw_output)
        if not question:
            print(f"[skip] no parseable question for dimension={dimension!r}: {raw_output!r}")
            continue
        print(f"  [{dimension}] question generated in {time.time()-t1:.1f}s: {question!r}", flush=True)

        print(f"\n  ---- [{dimension}] question vs. retrieved context ----", flush=True)
        print(f"  QUESTION: {question}", flush=True)
        for i, chunk in enumerate(retrieved_chunks):
            source = chunk.metadata.get("source", "unknown")
            preview = chunk.page_content.strip().replace("\n", " ")
            print(f"  CHUNK {i+1} (source: {source}): {preview}", flush=True)
        print("  " + "-" * 50 + "\n", flush=True)

        rows["user_input"].append(search_query)
        rows["response"].append(question)
        rows["retrieved_contexts"].append([c.page_content for c in retrieved_chunks])

    return Dataset.from_dict(rows)


# ---------------------------------------------------------------------------
# 5. Metrics.
#    - context_precision: were retrieved chunks useful for producing the
#      generated question?
#    - two AspectCritic checks tailored to this task's actual requirements.
# ---------------------------------------------------------------------------
def build_metrics(quick: bool = False):
    context_precision = LLMContextPrecisionWithoutReference()

    if quick:
        # Cheapest single metric -- one LLM call per sample, for a fast sanity check.
        return [context_precision]

    context_grounded = AspectCritic(
        name="context_grounded",
        definition=(
            "Return 1 if the submitted question could plausibly be answered "
            "making use of the information in the retrieved context, i.e. it is "
            "clearly connected to specific content in the context rather than "
            "a generic data-quality question that could apply to any dataset. "
            "Return 0 otherwise."
        ),
    )

    dimension_relevant = AspectCritic(
        name="dimension_relevant",
        definition=(
            "The retrieved context is relevant for assessing the data quality dimension "
            "as part of the user_input (after 'Focus:'). Return 1 if the "
            "submitted question is about that dimension. Return 0 "
            "if the question is off-topic or belongs to a different "
            "data-quality dimension."
        ),
    )

    return [context_precision, context_grounded, dimension_relevant]


# ---------------------------------------------------------------------------
# 6. Run.
# ---------------------------------------------------------------------------
def run_evaluation(dataset: Dataset, judge_llm, judge_embeddings, out_csv: str = OUT_CSV, quick: bool = False):
    metrics = build_metrics(quick=quick)

    # Single local model, one request at a time -> no concurrency.
    # llama.cpp generation is slow; give it room.
    run_config = RunConfig(max_workers=1, timeout=900)

    results = evaluate(
        dataset=dataset,
        metrics=metrics,
        llm=judge_llm,
        embeddings=judge_embeddings,
        run_config=run_config,
        raise_exceptions=False,
    )

    df = results.to_pandas()
    df.to_csv(out_csv, index=False)
    print(df)
    print("\nAggregate scores:")
    print(results)
    return df


if __name__ == "__main__":
    # Set QUICK_TEST=1 to run a fast 1-scenario, 1-metric sanity check instead
    # of the full evaluation, e.g.:
    #   QUICK_TEST=1 python evaluation/ragas_eval.py
    quick = os.environ.get("QUICK_TEST", "0") == "1"

    if not os.path.exists(VECTORSTORE_PATH):
        raise SystemExit(
            f"Vectorstore not found at '{VECTORSTORE_PATH}'. Run the Streamlit app once "
            "first to build it -- this script intentionally doesn't rebuild it."
        )

    print(f"[{'QUICK TEST' if quick else 'FULL RUN'}] Loading generation model (RAGChain)...", flush=True)
    t0 = time.time()
    rag_chain = RAGChain(vectorstore_path=VECTORSTORE_PATH, llama_model_path=MODEL_PATH)
    print(f"RAGChain loaded in {time.time()-t0:.1f}s", flush=True)

    scenarios = [
        {"dataset_info": "mammography data, 2000 DICOMS with csv patient metadata",
         "dimension": dim_name}
        for dim_name, _ in DIMENSIONS
    ]
    if quick:
        scenarios = scenarios[:1]

    print(f"Collecting {len(scenarios)} sample(s) by driving RAGChain directly...", flush=True)
    dataset = collect_eval_samples(rag_chain, scenarios)

    print("Loading judge LLM (second LlamaCpp instance -- see module docstring)...", flush=True)
    t1 = time.time()
    judge_llm = build_judge_llm()
    print(f"Judge LLM loaded in {time.time()-t1:.1f}s", flush=True)
    judge_embeddings = build_judge_embeddings()

    print("Running RAGAS evaluation...", flush=True)
    t2 = time.time()
    run_evaluation(dataset, judge_llm, judge_embeddings, quick=quick)
    print(f"Evaluation finished in {time.time()-t2:.1f}s", flush=True)