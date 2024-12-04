
SUMMARY_PROMPT = """Summarize the text below to help answer a question. 
    Do not directly answer the question, instead summarize 
    to give evidence to help answer the question. Include direct quotes. 
    Reply 'Not applicable' if text is irrelevant. 
    Use around 100 words. At the end of your response, provide a score from 1-5 
    indicating the relevance to the question. Output the score immediately following the answer 
    using the following format: Score: x/5. 
    Do not explain your score. 
    
    {context}
    Extracted from {citation}
    Question: {question}\n
    Relevant Information Summary:"""


QA_PROMPT = """Write an answer (about 250 words) 
    for the question below based on the provided context. 
    If the context provides insufficient information, reply 'I cannot answer'. 
    For each part of your answer, indicate which sources most support it 
    via valid citation markers at the end of sentences, like (LastName (Year)). 
    Answer in an unbiased, comprehensive, and scholarly tone. 
    If the question is subjective, provide an opinionated answer in the concluding 1-2 sentences. 
    Use Markdown for formatting code or text, and try to use direct quotes to support arguments.

    {context}
    Question: {question}
    Answer:"""

QA_CITATION_PROMPT = \
    "Write an answer (about 250 words) for the question below based on the provided context." \
    "Include relevant references. If the context provides insufficient information, reply 'I cannot " \
    "answer'. For each part of your answer, indicate which sources most support it via valid citation markers at " \
    "the end of sentences as shown in the Example. Reuse the same reference IDs as provided in the context. You do " \
    "not need to provide a references section in your response as it is already known by the user. Answer in an " \
    "unbiased, comprehensive, and scholarly tone. \n\n" \
    "###Example:\nIt is important to brush your teeth [REF-234]. You can get cavities [REF-654, REF-678]\n\n" \
    "###Question: {question}\n\n" \
    "###Context:\n{context}\n\n" \
    "###Answer:"
