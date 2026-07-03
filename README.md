# HF Docs Agent

An agentic RAG system for querying Hugging Face Transformers and PEFT documentation. Built with hybrid retrieval, cross-encoder reranking, and a LangGraph ReAct agent that decides between two tools per query.

Live demo: https://huggingface.co/spaces/mirha55an/hf-docs-agent

---

## Architecture

```
Query
  │
  ▼
LangGraph ReAct Agent
  │
  ├── retrieve()          BM25 + ChromaDB → RRF → bge-reranker-v2-m3
  │
  └── github_search()     GitHub REST API → source code snippets
  │
  ▼
Gemini / Llama Generation
  │
  ▼
Answer
```

**Retrieval stack:**

- Hybrid search: BM25 keyword search (`rank-bm25`) + vector search (`ChromaDB` + `bge-small-en`)
- Fusion: Reciprocal Rank Fusion (RRF)
- Reranking: `bge-reranker-v2-m3` cross-encoder on top-20 fused candidates

**Agent:**

- LangGraph ReAct loop — Thought → Action → Observation → repeat
- Two tools: `retrieve` for doc questions, `github_search` for implementation questions
- Answers directly for greetings and general knowledge without calling any tool

**Evaluation (RAGAS):**

| Metric            | Score  |
| ----------------- | ------ |
| Faithfulness      | 0.8076 |
| Answer Relevancy  | 0.9394 |
| Context Precision | 0.6389 |

---

## Stack

FastAPI · LangGraph · ChromaDB · sentence-transformers · rank-bm25 · bge-reranker-v2-m3 · RAGAS · Docker · HF Spaces

---

## Local Setup

### 1. Clone the repo

```bash
git clone https://github.com/mirha55an/hf-docs-agent
cd hf-docs-agent
```

### 2. Create and activate a virtual environment

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Set environment variables

Create a `.env` file in the project root:

```
GEMINI_API_KEY=your_gemini_key
GITHUB_TOKEN=your_github_token
GROQ_API_KEY=your_groq_key
```

Get your keys:

- Gemini: https://aistudio.google.com/apikey
- GitHub token: github.com → Settings → Developer Settings → Personal Access Tokens → Tokens (classic) → tick `public_repo` only
- Groq: https://console.groq.com/keys

### 5. Clone the Transformers docs

The ingestion script reads directly from the Transformers GitHub repo. Clone it into the project root:

```bash
git clone --depth 1 https://github.com/huggingface/transformers.git
```

`--depth 1` pulls only the latest commit, not the full history — much faster.

### 6. Build the indexes

This step chunks the docs, generates embeddings, builds the ChromaDB collection, and saves the BM25 index and chunks to disk. Run it once — you don't need to rerun it unless the docs change.

```bash
python -m src.ingestion
```

This creates three files:

```
data/
├── chroma_db/       # vector index (ChromaDB)
├── bm25.pkl         # keyword index
└── chunks.pkl       # chunk text and metadata
```

### 7. Run the app

```bash
uvicorn app:app --reload
```

Open http://localhost:7860 in your browser — you'll see the chat interface.

---

## Running Evaluation

To reproduce the RAGAS evaluation scores or measure the impact of changes to retrieval:

```bash
python -m eval.run_eval
```

This runs all 20 questions through the agent, collects retrieved contexts and answers, and scores them with RAGAS. Output is a results table printed to terminal.

To add questions or update ground truth answers, edit `eval/eval_dataset.py`.

---

## Docker

Build and run locally with Docker:

```bash
docker build -t hf-docs-agent .

docker run -p 7860:7860 \
  -e GEMINI_API_KEY=your_key \
  -e GITHUB_TOKEN=your_token \
  -e GROQ_API_KEY=your_key \
  hf-docs-agent
```

Note: the `data/` folder with built indexes must exist before building the Docker image. Run Step 6 first.

---
