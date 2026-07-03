from fastapi import FastAPI
from src.agent import run_langgraph_agent
from pydantic import BaseModel
from fastapi.responses import StreamingResponse

app = FastAPI(title="HF Docs Agent")

class QueryRequest(BaseModel):
    question: str
    
@app.get("/health")
def health_check():
    return {"status": "ok"}

@app.post("/ask")
def ask_question(request: QueryRequest):
    answer = run_langgraph_agent(request.question)
    return {"answer": answer}

@app.get("/")
def root():
    return {"message": "HF Docs Agent is running. POST to /ask with {question: ...}"}