---
title: HF Docs Agent
colorFrom: blue
colorTo: green
sdk: docker
pinned: false
---

# HF Docs Agent

An agentic RAG system for querying Hugging Face Transformers and PEFT documentation.

## Architecture

[paste the diagram from our earlier conversation]

## Retrieval Stack

- Hybrid search: BM25 (rank_bm25) + vector search (ChromaDB + bge-small-en)
- Fusion: Reciprocal Rank Fusion
- Reranking: bge-reranker-v2-m3 (cross-encoder)

## Agent

- LangGraph ReAct agent with two tools: docs retrieval and GitHub source lookup
- Decides per query whether to retrieve, search GitHub, or answer directly

## Evaluation (RAGAS)

| Metric            | Score  |
| ----------------- | ------ |
| Faithfulness      | 0.8076 |
| Answer Relevancy  | 0.9394 |
| Context Precision | 0.6389 |

## Stack

Python · FastAPI · LangGraph · ChromaDB · sentence-transformers · RAGAS · Docker · HF Spaces
