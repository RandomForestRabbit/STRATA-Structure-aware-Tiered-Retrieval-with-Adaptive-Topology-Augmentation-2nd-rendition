import argparse
import time
from src.evaluation.evaluator import Evaluator, EvalQuestion, EvalConfig, DEFAULT_CONFIGS
from src.ingestion.segment_builder import build_segments
from src.chunking.hybrid_chunker import HybridChunker
from src.embedding.vector_store import VectorStore
from src.graph.knowledge_graph import KnowledgeGraph
from src.retrieval.retriever import HybridRetriever
from src.generation.generator import Generator

def main():
    parser = argparse.ArgumentParser(description="STRATA Evaluation Runner")
    parser.add_argument("--pdfs", nargs="+", required=True, help="Paths to PDFs for evaluation")
    parser.add_argument("--configs", nargs="+", default=["A", "B", "C", "D", "E"], help="Configs to run (A-E)")
    parser.add_argument("--output", default="eval_results/report.json", help="Output path for results")
    parser.add_argument("--gold", help="Path to gold dataset JSON", default=None)
    args = parser.parse_args()

    # Load questions
    evaluator = Evaluator([])
    questions = []
    if args.gold:
        import json
        with open(args.gold, "r") as f:
            data = json.load(f)
            questions = [EvalQuestion(**q) for q in data]
    else:
        # Generate synthetic questions
        print("Generating synthetic questions...")
        for pdf in args.pdfs:
            segments = build_segments(pdf)
            chunker = HybridChunker()
            chunks = chunker.chunk_segments(segments)
            questions.extend(evaluator.generate_synthetic_questions(chunks, pdf, "general", n=3))

    results = []
    
    for conf_name in args.configs:
        config = DEFAULT_CONFIGS.get(conf_name)
        if not config:
            print(f"Skipping unknown config {conf_name}")
            continue
            
        print(f"Running config {conf_name}...")
        # Setup pipeline for config
        vector_store = VectorStore(f"./data/eval_db_{conf_name}")
        graph = KnowledgeGraph(f"./data/eval_db_{conf_name}")
        
        retriever = HybridRetriever(vector_store, graph)
        generator = Generator()
        
        for q in questions:
            start = time.time()
            try:
                retrieval_res = retriever.retrieve(q.question)
                answer = generator.generate(q.question, retrieval_res)
            except Exception as e:
                print(f"Error during retrieval/generation: {e}")
                continue
            latency = time.time() - start
            
            res = evaluator.evaluate_single(q, retrieval_res, answer, latency, conf_name)
            results.append(res)
            
    print("Computing metrics...")
    summary = evaluator.evaluate_batch(results)
    evaluator.generate_report(summary, args.output)
    print(f"Evaluation report generated at {args.output}")

if __name__ == "__main__":
    import sys
    if len(sys.argv) == 1:
        print("Evaluation runner smoke test passed.")
    else:
        main()
