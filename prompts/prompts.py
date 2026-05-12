from langchain_core.prompts import PromptTemplate

QUESTIONNAIRE_PROMPT = PromptTemplate(
    input_variables=["context_chunks", "dataset_info", "generated_questions", "dimension"], 
    template="""<|im_start|>system
    You are a data quality expert generating questionnaire questions. Respond only with <|im_start|>user
    You evaluate the quality of a dataset on the dimension "{dimension}"
    (The extent to which data values match real world knowledge.).

    Generate exactly ONE yes/no question based on this context:
    {context_chunks}

    Dataset information:
    {dataset_info}

    Already generated questions (do not repeat):
    {generated_questions}

    Collected information on the datset:
    {dataset_info}

    Output only in this format:
    Next question : <your yes/no question>
    <|im_end|>
    <|im_start|>assistant
    <think>

   </think>
    """
)
