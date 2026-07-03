import os, pickle, re
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi
import chromadb

DOCS_PATH = 'transformers/docs/source/en'
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CHROMA_PATH = os.path.join(BASE_DIR, "data", "chroma_db")
BM25_PATH = os.path.join(BASE_DIR, "data", "bm25.pkl")
CHUNKS_PATH = os.path.join(BASE_DIR, "data", "chunks.pkl")
MAX_CHUNK_CHARS = 1500


def tokenize(text):
    # lowercase, strip markdown symbols and punctuation, split on whitespace
    text = text.lower()
    text = re.sub(r'[`\[\](){}.,;:!?"\'-]', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip().split()

def split_chunk(text: str, max_chars: int = MAX_CHUNK_CHARS) -> list:
    if len(text) <= max_chars:
        return [text]
    paragraphs = text.split('\n\n')
    result, current = [], ""
    for para in paragraphs:
        if len(current) + len(para) < max_chars:
            current += "\n\n" + para
        else:
            if current:
                result.append(current.strip())
            current = para
    if current:
        result.append(current.strip())
    return result

def load_chunks():
    chunks= []
    if os.path.exists(DOCS_PATH):
        for root, dirs, files in os.walk(DOCS_PATH):
            for file in files:
                if file.endswith(('.md', '.mdx')):
                    filepath = os.path.join(root,file)
                    with open(filepath, 'r', encoding='utf-8') as f:
                        text = f.read()
                        # split into sections
                        import re
                        sections = re.split(r'\n#{1,3} ', text)

                        for section in sections:
                            section = section.strip()
                            if len(section) > 200:
                                for sub_chunk in split_chunk(section):
                                    if len(sub_chunk) > 200:
                                        chunks.append({
                                            "id" : f"{filepath}-{len(chunks)}",
                                            "text" : sub_chunk,
                                            "source" : filepath
                                        })

    return chunks

def build_indexes(chunks):
    # Embeddings
    model = SentenceTransformer('BAAI/bge-small-en')
    texts = [c['text'] for c in chunks]
    embeddings = model.encode(texts, show_progress_bar=True)
    
    # ChromaDB
    chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)
    collection = chroma_client.get_or_create_collection(name="transformers")

    collection.add(
        ids = [c['id'] for c in chunks],
        documents = [c['text'] for c in chunks],
        metadatas = [{'source' : c['source'], 'index' : i} for i, c in enumerate(chunks)],
        embeddings = embeddings
    )
    
    # BM25
    tokenized_chunks = [tokenize(c["text"]) for c in chunks]
    bm25 = BM25Okapi(tokenized_chunks)
    
    with open(BM25_PATH, "wb") as f:
        pickle.dump(bm25, f)
    with open(CHUNKS_PATH, "wb") as f:
        pickle.dump(chunks, f)

    print(f"Indexed {len(chunks)} chunks.")
    
if __name__ == "__main__":
    chunks = load_chunks()
    build_indexes(chunks)