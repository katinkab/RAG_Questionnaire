from langchain_core.prompts import PromptTemplate

QUESTIONNAIRE_PROMPT = PromptTemplate(
    input_variables=["context_chunks", "dataset_info", "generated_questions", "dimension"], 
    template="""<|im_start|>system
    You are a data quality expert generating questionnaire questions. Respond only with <|im_start|>user
    You evaluate the quality of a dataset on the dimension "{dimension}"
    (The extent to which data values match real world knowledge.).

    Use the context below as inspiration to generate exactly ONE yes/no question.
    Reformulate the question to be specific to the dataset described above.
    Use the dataset type and characteristics in your question rather than generic terms.
    Context:
    {context_chunks}

    Dataset information:
    {dataset_info}

    Already generated questions (do not repeat):
    {generated_questions}

    Collected information on the datset:
    {dataset_info}

    Output only in this format:
    Next question : <your yes/no question>
    Source: <name of the source document the question is based on>
    <|im_end|>
    <|im_start|>assistant
    <think>

   </think>
    """
)
