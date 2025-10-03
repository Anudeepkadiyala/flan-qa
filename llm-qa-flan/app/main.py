from fastapi import FastAPI
from pydantic import BaseModel, Field
from app.inference import QAEngine

app = FastAPI(title="FLAN-T5 QA API", version="1.0.0")
engine = QAEngine()

class QARequest(BaseModel):
    context: str = Field(..., description="Context paragraph")
    question: str = Field(..., description="Question to answer")

class QAResponse(BaseModel):
    answer: str

@app.get("/healthz")
def health():
    return {"status": "ok"}

@app.post("/qa", response_model=QAResponse)
def qa(req: QARequest):
    ans = engine.answer(req.context, req.question)
    return QAResponse(answer=ans)
