import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""  # Force CPU (prevents GPU-related startup issues)

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
import shutil

app = FastAPI(title="AI Career Copilot API")

# Ensure static folder exists
os.makedirs("static", exist_ok=True)

# Serve static frontend
app.mount("/static", StaticFiles(directory="static"), name="static")


# Root endpoint → serves UI
@app.get("/")
async def serve_ui():
    return FileResponse("static/index.html")


# Request model
class AnswerPayload(BaseModel):
    question: str
    user_answer: str


# 🔹 Analyze resume + JD
@app.post("/api/analyze")
async def analyze_profile(jd_text: str = Form(...), resume: UploadFile = File(...)):
    
    # Lazy imports (CRITICAL for deployment)
    from analyzer import analyze_resume_against_jd, extract_text_from_pdf
    from rag_engine import setup_rag_retriever

    file_location = f"temp_{resume.filename}"

    with open(file_location, "wb+") as file_object:
        shutil.copyfileobj(resume.file, file_object)

    try:
        analysis = analyze_resume_against_jd(file_location, jd_text)
        resume_text = extract_text_from_pdf(file_location)

        # Persist retriever
        setup_rag_retriever(resume_text, jd_text)

    finally:
        if os.path.exists(file_location):
            os.remove(file_location)

    return {
        "status": "success",
        "analysis": analysis
    }


# 🔹 Generate interview questions
@app.get("/api/generate-questions")
async def get_questions(focus: str = "Technical Skills"):

    # Lazy imports
    from rag_engine import load_rag_retriever, generate_interview_questions

    try:
        retriever = load_rag_retriever()
    except Exception:
        raise HTTPException(
            status_code=400,
            detail="No resume data found. Please call /api/analyze first."
        )

    questions = generate_interview_questions(retriever, focus)

    return {
        "questions": questions
    }


# 🔹 Evaluate answer
@app.post("/api/evaluate")
async def evaluate_answer(payload: AnswerPayload):

    # Lazy import
    from evaluator import evaluate_candidate_answer

    evaluation = evaluate_candidate_answer(
        payload.question,
        payload.user_answer
    )

    return {
        "evaluation": evaluation
    }