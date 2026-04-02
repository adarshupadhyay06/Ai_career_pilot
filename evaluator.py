from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from models import EvaluationResult
from dotenv import load_dotenv

load_dotenv()

llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)


def evaluate_candidate_answer(question: str, user_answer: str):
    structured_llm = llm.with_structured_output(EvaluationResult)

    prompt = PromptTemplate(
        template="""You are a strict but fair technical interviewer.
        Evaluate the candidate's answer to the following question.
        
        Question: {question}
        Candidate Answer: {answer}
        
        Score the answer out of 10. Provide constructive feedback on what they missed 
        or did well, and provide a perfect ideal answer.""",
        input_variables=["question", "answer"]
    )

    chain = prompt | structured_llm
    return chain.invoke({"question": question, "answer": user_answer})