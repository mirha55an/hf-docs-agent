import chromadb
import pickle
from sentence_transformers import SentenceTransformer, CrossEncoder
from src.ingestion import tokenize
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # project root

CHROMA_PATH = os.path.join(BASE_DIR, "data", "chroma_db")
BM25_PATH = os.path.join(BASE_DIR, "data", "bm25.pkl")
CHUNKS_PATH = os.path.join(BASE_DIR, "data", "chunks.pkl")

embed_model = SentenceTransformer('BAAI/bge-small-en')
reranker = CrossEncoder('BAAI/bge-reranker-v2-m3')



chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)
collection = chroma_client.get_collection(name="transformers")

with open(BM25_PATH, "rb") as f:
    bm25 = pickle.load(f)
with open(CHUNKS_PATH, "rb") as f:
    chunks = pickle.load(f)
    
def vector_search(query_embedding, top_k=15):
    return collection.query(
    query_embeddings = [query_embedding],
    n_results = top_k,
    include=['documents', 'metadatas', 'embeddings']
    )
    
def bm25_search(query, top_k=15):
  tokenized_query = tokenize(query)
  scores = bm25.get_scores(tokenized_query)

  top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]

  return [(idx, scores[idx]) for idx in top_indices]\
      
      
def reciprocal_rank_fusion(bm25_results, vector_results, k=60, top_k=5):
  scores = {}

  for rank, (idx, _) in enumerate(bm25_results):
    scores[idx] = scores.get(idx,0) + 1 / (k + rank + 1)

  for rank, meta in enumerate(vector_results['metadatas'][0]):
    idx = meta['index']
    scores[idx] = scores.get(idx,0) + 1 / (k + rank + 1)

  top_indices = sorted(scores, key=lambda i : scores[i], reverse=True)[:top_k]

  return top_indices

def rerank(query, candidate_indices, top_k=15, threshold: float = 0.1):
  pairs = [(query, chunks[idx]['text']) for idx in candidate_indices]
  scores = reranker.predict(pairs)

  scored = sorted(zip(candidate_indices, scores), key=lambda x: x[1], reverse=True)

  filtered = [(idx, score) for idx, score in scored if score > threshold]
  return filtered[:top_k] if filtered else scored[:1]

def retrieve (question: str) -> str:
  question_embed = embed_model.encode(question)
  bm25_results = bm25_search(question, top_k=20)
  vector_results = vector_search(question_embed, top_k=15)

  top_indices = reciprocal_rank_fusion(bm25_results, vector_results)

  reranked = rerank(question, top_indices, top_k=3)

  top_matches = [chunks[i] for i,score in reranked]

  return '\n\n'.join(f"Source: {m['source']}\n{m['text']}" for m in top_matches)