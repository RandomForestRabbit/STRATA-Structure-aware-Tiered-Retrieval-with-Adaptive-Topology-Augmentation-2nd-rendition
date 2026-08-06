import json
import os
from dataclasses import dataclass
from typing import List, Dict, Any, Optional

import anthropic
from src.retrieval.retriever import RetrievalResult

@dataclass
class EvalQuestion:
    question: str
    gold_answer: str
    gold_chunk_ids: List[str]       # IDs of chunks that contain the answer
    source_doc_ids: List[str]       # which docs the answer comes from
    is_cross_doc: bool = False      # requires info from multiple docs
    pdf_type: str = 'general'       # 'tabular' | 'image_heavy' | 'long_form' | 'instructional'

@dataclass
class EvalConfig:
    name: str                        # 'A' through 'E'
    description: str
    use_hybrid_chunking: bool = True
    use_graph_retrieval: bool = True
    use_reranker: bool = False
    max_chunk_tokens: int = 384
    vector_k: int = 15
    graph_hops: int = 2
    alpha: float = 0.6

DEFAULT_CONFIGS = {
    'A': EvalConfig('A', 'Baseline: fixed-size chunking, vector-only', use_hybrid_chunking=False, use_graph_retrieval=False),
    'B': EvalConfig('B', 'Hybrid chunking + vector-only', use_hybrid_chunking=True, use_graph_retrieval=False),
    'C': EvalConfig('C', 'Fixed chunking + graph retrieval', use_hybrid_chunking=False, use_graph_retrieval=True),
    'D': EvalConfig('D', 'Full: hybrid chunking + graph retrieval', use_hybrid_chunking=True, use_graph_retrieval=True),
    'E': EvalConfig('E', 'Full + cross-encoder rerank', use_hybrid_chunking=True, use_graph_retrieval=True, use_reranker=True),
}

class Evaluator:
    def __init__(self, questions: List[EvalQuestion]):
        self.questions = questions

    def recall_at_k(self, retrieved_chunk_ids: List[str], gold_chunk_ids: List[str], k: int = 10) -> float:
        if not gold_chunk_ids:
            return 1.0
        retrieved_k = retrieved_chunk_ids[:k]
        hits = sum(1 for gid in gold_chunk_ids if gid in retrieved_k)
        return hits / len(gold_chunk_ids)

    def mrr(self, retrieved_chunk_ids: List[str], gold_chunk_ids: List[str]) -> float:
        if not gold_chunk_ids:
            return 1.0
        for i, rid in enumerate(retrieved_chunk_ids):
            if rid in gold_chunk_ids:
                return 1.0 / (i + 1)
        return 0.0

    def answer_correctness(self, predicted: str, gold: str, model: str = 'claude-3-5-sonnet-20241022') -> float:
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            return -1.0
        
        try:
            client = anthropic.Anthropic(api_key=api_key)
            prompt = f"Gold answer: {gold}\nPredicted answer: {predicted}\nRate the correctness of the predicted answer compared to the gold answer on a scale of 0.0 to 1.0. Reply with ONLY a number."
            
            response = client.messages.create(
                model=model,
                max_tokens=10,
                messages=[{"role": "user", "content": prompt}]
            )
            score_text = response.content[0].text.strip()
            return float(score_text)
        except Exception:
            return -1.0

    def evaluate_single(self, question: EvalQuestion, retrieval_result: RetrievalResult, generated_answer: str, latency: float, config: str) -> dict:
        retrieved_ids = [chunk.chunk_id for chunk in retrieval_result.chunks]
        
        recall_5 = self.recall_at_k(retrieved_ids, question.gold_chunk_ids, k=5)
        recall_10 = self.recall_at_k(retrieved_ids, question.gold_chunk_ids, k=10)
        mrr_score = self.mrr(retrieved_ids, question.gold_chunk_ids)
        correctness = self.answer_correctness(generated_answer, question.gold_answer)
        
        return {
            "question": question.question,
            "recall_5": recall_5,
            "recall_10": recall_10,
            "mrr": mrr_score,
            "answer_correctness": correctness,
            "latency": latency,
            "config": config,
            "pdf_type": question.pdf_type
        }

    def evaluate_batch(self, results: List[dict]) -> dict:
        overall = {"recall_5": 0.0, "recall_10": 0.0, "mrr": 0.0, "answer_correctness": 0.0, "latency": 0.0, "count": 0}
        by_pdf_type = {}
        by_config = {}
        
        for r in results:
            overall["count"] += 1
            for k in ["recall_5", "recall_10", "mrr", "latency"]:
                overall[k] += r[k]
            if r["answer_correctness"] >= 0:
                if "corr_count" not in overall:
                    overall["corr_count"] = 0
                    overall["corr_sum"] = 0.0
                overall["corr_count"] += 1
                overall["corr_sum"] += r["answer_correctness"]
                
            p_type = r["pdf_type"]
            if p_type not in by_pdf_type:
                by_pdf_type[p_type] = {"recall_5": 0.0, "recall_10": 0.0, "mrr": 0.0, "answer_correctness": 0.0, "latency": 0.0, "count": 0, "corr_count": 0, "corr_sum": 0.0}
            by_pdf_type[p_type]["count"] += 1
            for k in ["recall_5", "recall_10", "mrr", "latency"]:
                by_pdf_type[p_type][k] += r[k]
            if r["answer_correctness"] >= 0:
                by_pdf_type[p_type]["corr_count"] += 1
                by_pdf_type[p_type]["corr_sum"] += r["answer_correctness"]

            conf = r["config"]
            if conf not in by_config:
                by_config[conf] = {"recall_5": 0.0, "recall_10": 0.0, "mrr": 0.0, "answer_correctness": 0.0, "latency": 0.0, "count": 0, "corr_count": 0, "corr_sum": 0.0}
            by_config[conf]["count"] += 1
            for k in ["recall_5", "recall_10", "mrr", "latency"]:
                by_config[conf][k] += r[k]
            if r["answer_correctness"] >= 0:
                by_config[conf]["corr_count"] += 1
                by_config[conf]["corr_sum"] += r["answer_correctness"]
                
        def _avg(d):
            count = d["count"]
            if count == 0: return d
            res = {k: d[k]/count for k in ["recall_5", "recall_10", "mrr", "latency"]}
            if d.get("corr_count", 0) > 0:
                res["answer_correctness"] = d["corr_sum"] / d["corr_count"]
            else:
                res["answer_correctness"] = -1.0
            return res
            
        return {
            "overall": _avg(overall),
            "by_pdf_type": {k: _avg(v) for k, v in by_pdf_type.items()},
            "by_config": {k: _avg(v) for k, v in by_config.items()}
        }

    def generate_report(self, results: dict, output_path: str):
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        md_content = "# Evaluation Report\n\n"
        md_content += "## Overall Metrics\n"
        md_content += f"- Recall@5: {results['overall']['recall_5']:.4f}\n"
        md_content += f"- Recall@10: {results['overall']['recall_10']:.4f}\n"
        md_content += f"- MRR: {results['overall']['mrr']:.4f}\n"
        md_content += f"- Answer Correctness: {results['overall']['answer_correctness']:.4f}\n"
        md_content += f"- Latency: {results['overall']['latency']:.4f}s\n\n"
        
        md_content += "## Metrics by Config\n"
        for conf, metrics in results['by_config'].items():
            md_content += f"### Config {conf}\n"
            for k, v in metrics.items():
                md_content += f"- {k}: {v:.4f}\n"
            md_content += "\n"
            
        md_path = output_path.replace(".json", ".md") if output_path.endswith(".json") else output_path + ".md"
        with open(md_path, "w", encoding="utf-8") as f:
            f.write(md_content)
            
        json_path = output_path.replace(".md", ".json") if output_path.endswith(".md") else output_path + ".json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)

    def generate_synthetic_questions(self, chunks: List[Any], doc_id: str, pdf_type: str, n: int = 5) -> List[EvalQuestion]:
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key or len(chunks) == 0:
            return [EvalQuestion(f"What is {chunk.text[:20]}?", chunk.text, [chunk.chunk_id], [doc_id], False, pdf_type) for chunk in chunks[:n]]
            
        try:
            client = anthropic.Anthropic(api_key=api_key)
            context = "\n".join([f"[{chunk.chunk_id}] {chunk.text}" for chunk in chunks[:min(20, len(chunks))]])
            prompt = f"Given the following chunks of text:\n{context}\nGenerate {n} questions that can be answered using this text. Provide the output in JSON format with fields: 'question', 'gold_answer', and 'gold_chunk_ids' (list of chunk IDs used)."
            
            response = client.messages.create(
                model="claude-3-5-sonnet-20241022",
                max_tokens=1000,
                messages=[{"role": "user", "content": prompt}]
            )
            try:
                data = json.loads(response.content[0].text)
                return [EvalQuestion(q["question"], q["gold_answer"], q["gold_chunk_ids"], [doc_id], False, pdf_type) for q in data]
            except:
                pass
        except Exception:
            pass
            
        return [EvalQuestion(f"What is {chunk.text[:20]}?", chunk.text, [chunk.chunk_id], [doc_id], False, pdf_type) for chunk in chunks[:n]]

if __name__ == "__main__":
    evaluator = Evaluator([])
    print("Evaluator module smoke test passed.")
