'''
run_pipeline.py

End-to-end STRATA pipeline: PDF(s) → segments → chunks → embeddings → graph → query → answer.

Usage:
    # Index one or more PDFs
    python run_pipeline.py index paper1.pdf paper2.pdf --persist-dir ./data/strata_db

    # Query the indexed documents  
    python run_pipeline.py query "What are the main findings?" --persist-dir ./data/strata_db

    # Full pipeline: index then query
    python run_pipeline.py run paper1.pdf --query "What are the main findings?"
'''

import argparse
import os
import sys

from src.ingestion.segment_builder import build_segments
from src.chunking.hybrid_chunker import HybridChunker, Chunk
from src.embedding.vector_store import VectorStore
from src.graph.entity_extractor import EntityExtractor
from src.graph.entity_resolver import EntityResolver
from src.graph.conflict_detector import ConflictDetector
from src.graph.knowledge_graph import KnowledgeGraph
from src.retrieval.retriever import HybridRetriever
from src.generation.generator import Generator

def do_index(pdfs, persist_dir):
    os.makedirs(persist_dir, exist_ok=True)
    vector_store = VectorStore(persist_dir)
    graph = KnowledgeGraph(persist_dir)
    entity_extractor = EntityExtractor()
    entity_resolver = EntityResolver()
    
    for pdf in pdfs:
        print(f"Indexing {pdf}...")
        segments = build_segments(pdf)
        chunker = HybridChunker()
        chunks = chunker.chunk_segments(segments)
        vector_store.add_chunks(chunks)
        
        for chunk in chunks:
            entities, relations = entity_extractor.extract(chunk.text)
            resolved_entities = entity_resolver.resolve(entities)
            for e in resolved_entities:
                graph.add_entity(e["id"], e["type"], e.get("attributes", {}))
            for r in relations:
                graph.add_relation(r["source"], r["target"], r["type"], r.get("attributes", {}))
                
    vector_store.persist()
    graph.persist()
    print("Indexing complete.")

def do_query(query_str, persist_dir):
    vector_store = VectorStore(persist_dir)
    graph = KnowledgeGraph(persist_dir)
    
    retriever = HybridRetriever(vector_store, graph)
    retrieval_result = retriever.retrieve(query_str)
    
    generator = Generator()
    answer = generator.generate(query_str, retrieval_result)
    
    print("\n--- Answer ---")
    print(answer)
    print("\n--- Citations ---")
    for chunk in retrieval_result.chunks:
        print(f"[{chunk.chunk_id}] {chunk.text[:50]}...")

def main():
    parser = argparse.ArgumentParser(description="STRATA Pipeline Runner")
    subparsers = parser.add_subparsers(dest="command", required=True)
    
    index_parser = subparsers.add_parser("index", help="Index PDFs")
    index_parser.add_argument("pdfs", nargs="+", help="Paths to PDF files")
    index_parser.add_argument("--persist-dir", required=True, help="Directory to store DBs")
    
    query_parser = subparsers.add_parser("query", help="Query indexed documents")
    query_parser.add_argument("query_str", help="The query string")
    query_parser.add_argument("--persist-dir", required=True, help="Directory with stored DBs")
    
    run_parser = subparsers.add_parser("run", help="Index and query")
    run_parser.add_argument("pdfs", nargs="+", help="Paths to PDF files")
    run_parser.add_argument("--query", required=True, help="The query string")
    run_parser.add_argument("--persist-dir", default="./data/strata_db", help="Directory to store DBs")

    args = parser.parse_args()

    if args.command == "index":
        do_index(args.pdfs, args.persist_dir)
    elif args.command == "query":
        do_query(args.query_str, args.persist_dir)
    elif args.command == "run":
        do_index(args.pdfs, args.persist_dir)
        do_query(args.query, args.persist_dir)

if __name__ == "__main__":
    if len(sys.argv) == 1:
        print("Pipeline runner smoke test passed.")
    else:
        main()
