import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from models import ResumeAnalysis
from dotenv import load_dotenv

load_dotenv()

# Initialize Groq (completely free, no billing needed)
llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0,
    max_retries=2
)


def extract_text_from_pdf(file_path: str) -> str:
    loader = PyPDFLoader(file_path)
    pages = loader.load()
    return "\n".join([page.page_content for page in pages])


def analyze_resume_against_jd(resume_path: str, jd_text: str):
    resume_text = extract_text_from_pdf(resume_path)

    structured_llm = llm.with_structured_output(ResumeAnalysis)

    prompt = PromptTemplate(
        template="""You are an expert technical recruiter. 
        Compare the following Resume with the Job Description.
        
        Job Description:
        {jd}
        
        Resume:
        {resume}
        
        Provide a match score, list strengths, identify missing skills (weaknesses), 
        and suggest improved bullet points for the resume.""",
        input_variables=["jd", "resume"]
    )

    chain = prompt | structured_llm
    result = chain.invoke({"jd": jd_text, "resume": resume_text})

    return result