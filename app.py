from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, PlainTextResponse
from src.agent import run_langgraph_agent
from pydantic import BaseModel
from collections import deque
import time

app = FastAPI(title="HF Docs Agent")
app.mount("/static", StaticFiles(directory="static"), name="static")

# Store the last 200 log entries in memory for web viewing
log_buffer = deque(maxlen=200)

def log_event(message: str):
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    entry = f"[{timestamp}] {message}"
    print(entry, flush=True)
    log_buffer.append(entry)

class QueryRequest(BaseModel):
    question: str
    
@app.get("/health")
def health_check():
    return {"status": "ok"}

@app.post("/ask")
def ask_question(request: QueryRequest):
    log_event(f"QUESTION RECEIVED: {request.question}")
    t0 = time.time()
    try:
        answer = run_langgraph_agent(request.question)
        duration = time.time() - t0
        log_event(f"QUERY SUCCESS ({duration:.2f}s): Answer length {len(answer)} chars")
        return {"answer": answer}
    except Exception as e:
        log_event(f"QUERY ERROR: {e}")
        raise

@app.get("/logs")
def get_logs():
    content = "\n".join(log_buffer) if log_buffer else "No logs recorded yet."
    return PlainTextResponse(content)

@app.get("/")
def root():
    return FileResponse("static/index.html")