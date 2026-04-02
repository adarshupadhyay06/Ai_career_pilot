from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
import shutil
import os

from analyzer import analyze_resume_against_jd, extract_text_from_pdf
from rag_engine import setup_rag_retriever, load_rag_retriever, generate_interview_questions
from evaluator import evaluate_candidate_answer

app = FastAPI(title="AI Career Copilot API")

# Serve static files (index.html) from the "static" folder
os.makedirs("static", exist_ok=True)
app.mount("/static", StaticFiles(directory="static"), name="static")


# Serve the UI at root URL: http://localhost:8000/
@app.get("/")
async def serve_ui():
    return FileResponse("static/index.html")


class AnswerPayload(BaseModel):
    question: str
    user_answer: str


@app.post("/api/analyze")
async def analyze_profile(jd_text: str = Form(...), resume: UploadFile = File(...)):
    file_location = f"temp_{resume.filename}"

    with open(file_location, "wb+") as file_object:
        shutil.copyfileobj(resume.file, file_object)

    # Fixed: use try/finally so temp file is always deleted even if an error occurs
    try:
        analysis = analyze_resume_against_jd(file_location, jd_text)
        resume_text = extract_text_from_pdf(file_location)
        # Fixed: persist the retriever to disk instead of storing in memory dict
        setup_rag_retriever(resume_text, jd_text)
    finally:
        if os.path.exists(file_location):
            os.remove(file_location)

    return {"status": "success", "analysis": analysis}


@app.get("/api/generate-questions")
async def get_questions(focus: str = "Technical Skills"):
    # Fixed: load retriever from persisted disk store — works across workers and restarts
    try:
        retriever = load_rag_retriever()
    except Exception:
        raise HTTPException(
            status_code=400,
            detail="No resume data found. Please call /api/analyze first."
        )

    questions = generate_interview_questions(retriever, focus)
    return {"questions": questions}


@app.post("/api/evaluate")
async def evaluate_answer(payload: AnswerPayload):
    evaluation = evaluate_candidate_answer(payload.question, payload.user_answer)
    return {"evaluation": evaluation}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)