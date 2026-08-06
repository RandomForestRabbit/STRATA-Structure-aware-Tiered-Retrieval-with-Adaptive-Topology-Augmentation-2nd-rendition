import os
import uuid
import logging
from typing import List, Dict, Any, Optional

try:
    import chromadb
    from chromadb.config import Settings
except ImportError:
    logging.warning("chromadb is not installed. Please install it.")

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    logging.warning("sentence-transformers is not installed. Please install it.")

from src.chunking.hybrid_chunker import Chunk

class VectorStore:
    def __init__(self, collection_name='strata_chunks', persist_dir='./data/chroma_db',
                 model_name='all-MiniLM-L6-v2', batch_size=64):
        self.collection_name = collection_name
        self.persist_dir = persist_dir
        self.batch_size = batch_size
        
        os.makedirs(self.persist_dir, exist_ok=True)
        
        self.client = chromadb.PersistentClient(path=self.persist_dir)
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"}
        )
        
        try:
            self.embedder = SentenceTransformer(model_name)
        except Exception as e:
            raise RuntimeError(f"Failed to load sentence-transformers model {model_name}: {e}")

    def get_embedding(self, text: str) -> List[float]:
        embedding = self.embedder.encode(text, normalize_embeddings=True)
        return embedding.tolist()

    def get_embeddings(self, texts: List[str]) -> List[List[float]]:
        embeddings = self.embedder.encode(texts, normalize_embeddings=True)
        return embeddings.tolist()

    def add_chunks(self, chunks: List[Chunk]):
        if not chunks:
            return
            
        for i in range(0, len(chunks), self.batch_size):
            batch = chunks[i:i + self.batch_size]
            
            ids = [c.chunk_id for c in batch]
            documents = [c.text for c in batch]
            
            metadatas = []
            for c in batch:
                meta = {
                    "doc_id": c.doc_id,
                    "page_start": c.page_start,
                    "page_end": c.page_end,
                    "chunk_type": c.chunk_type,
                    "chunk_id": c.chunk_id
                }
                if c.metadata:
                    meta.update(c.metadata)
                metadatas.append(meta)
                
            embeddings = self.get_embeddings(documents)
            
            self.collection.upsert(
                ids=ids,
                documents=documents,
                embeddings=embeddings,
                metadatas=metadatas
            )

    def query(self, query_text: str, n_results: int = 10, where_filter: dict = None) -> List[dict]:
        query_embedding = self.get_embedding(query_text)
        return self.query_by_embedding(query_embedding, n_results=n_results, where_filter=where_filter)

    def query_by_embedding(self, embedding: List[float], n_results: int = 10, where_filter: dict = None) -> List[dict]:
        kwargs = {
            "query_embeddings": [embedding],
            "n_results": n_results,
            "include": ["documents", "metadatas", "distances"]
        }
        if where_filter:
            kwargs["where"] = where_filter
            
        results = self.collection.query(**kwargs)
        
        formatted_results = []
        if not results['ids'] or not results['ids'][0]:
            return formatted_results
            
        # Chroma returns lists of lists for multiple queries. We only did one query.
        for i in range(len(results['ids'][0])):
            formatted_results.append({
                "chunk_id": results['ids'][0][i],
                "text": results['documents'][0][i],
                # We use cosine space, distance is 1 - cosine similarity. Convert back to score.
                "score": 1.0 - results['distances'][0][i] if 'distances' in results and results['distances'] else None,
                "metadata": results['metadatas'][0][i]
            })
            
        # Sort by score descending
        formatted_results.sort(key=lambda x: x['score'] if x['score'] is not None else -1, reverse=True)
        return formatted_results

    def delete_collection(self):
        try:
            self.client.delete_collection(name=self.collection_name)
        except ValueError:
            pass # Collection doesn't exist


if __name__ == "__main__":
    import sys
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
    
    # Smoke test
    from src.chunking.hybrid_chunker import Chunk
    
    store = VectorStore(collection_name='smoke_test', persist_dir='./data/test_chroma_db')
    
    c1 = Chunk(
        chunk_id="doc1_chunk1",
        doc_id="doc1",
        text="The quick brown fox jumps over the lazy dog.",
        segments=[],
        page_start=1,
        page_end=1,
        chunk_type="prose",
        metadata={}
    )
    c2 = Chunk(
        chunk_id="doc1_chunk2",
        doc_id="doc1",
        text="Artificial intelligence and machine learning are revolutionizing technology.",
        segments=[],
        page_start=2,
        page_end=2,
        chunk_type="prose",
        metadata={"custom": "val"}
    )
    
    store.add_chunks([c1, c2])
    
    res = store.query("Tell me about AI", n_results=1)
    print("Query results for 'Tell me about AI':")
    for r in res:
        print(f"ID: {r['chunk_id']}, Score: {r['score']:.4f}\nText: {r['text']}")
        
    store.delete_collection()
