from typing import Literal, Dict
from pydantic import BaseModel, Field

class QuizQuestion(BaseModel):
    question: str = Field(description="The quiz question prompt")
    options: Dict[str, str] = Field(
        description="Dictionary mapping option keys (A, B, C, D) to option text"
    )
    correct_answer: str = Field(
        description="The key corresponding to the correct option, e.g., 'A', 'B', 'C', or 'D'"
    )
    explanation: str = Field(description="Explanation of why the correct option is right")
    difficulty: Literal["easy", "medium", "hard"] = Field(
        default="medium", description="Difficulty level of the question"
    )

class GradeResult(BaseModel):
    correct: bool = Field(description="True if the student's answer is correct, False otherwise")
    feedback: str = Field(description="Constructive feedback explaining why the answer is correct or incorrect")
    correct_answer: str = Field(description="The correct answer text or option")
