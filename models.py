from pydantic import BaseModel, Field
from typing import List


class ResumeAnalysis(BaseModel):
    score: int = Field(ge=0, le=100, description="Score out of 100 based on JD match")
    strengths: List[str] = Field(description="Key strengths found in the resume")
    weaknesses: List[str] = Field(description="Missing skills or weak bullet points")
    improved_bullets: List[str] = Field(description="Suggested rewrites for weak bullet points using action verbs")


class EvaluationResult(BaseModel):
    score: int = Field(ge=0, le=10, description="Score out of 10 for the user's interview answer")
    feedback: str = Field(description="Constructive feedback on the answer")
    ideal_answer: str = Field(description="An example of a perfect answer to the question")