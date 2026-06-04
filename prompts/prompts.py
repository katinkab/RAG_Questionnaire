from langchain_core.prompts import PromptTemplate

QUESTIONNAIRE_PROMPT = PromptTemplate(
    input_variables=["context_chunks", "dataset_info", "generated_questions", "dimension"],
    template="""<|im_start|>system
You are a data quality expert generating questionnaire questions for AI medical datasets.<|im_end|>
<|im_start|>user
You are assessing the data quality dimension "{dimension}" for the following dataset:
{dataset_info}

Use the topics below as inspiration to identify what area to ask about next:
{context_chunks}

Generate exactly ONE yes/no question that:
- Is specific to the dataset described above
- Covers a topic inspired by the context
- Has not been asked before (see history below)
- Uses natural medical language appropriate for the dataset

Already asked (do NOT repeat or ask anything with similar meaning):
{generated_questions}

Output only:
Next question : <your yes/no question>
Source: <name of the source document from the context that inspired this question>
<|im_end|>
<|im_start|>assistant
<think>

</think>

"""
)
