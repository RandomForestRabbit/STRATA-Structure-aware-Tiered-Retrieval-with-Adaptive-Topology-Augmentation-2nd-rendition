import logging
from dataclasses import dataclass
from typing import List, Dict, Any, Optional
import tiktoken
try:
    from sentence_transformers import CrossEncoder
except ImportError:
    CrossEncoder = None

logger = logging.getLogger(__name__)

@dataclass
class RetrievalResult:
    chunks: List[dict]
    query: str
    vector_results_count: int
    graph_results_count: int
    fused_count: int
    reranked: bool


class HybridRetriever:
    def __init__(self, vector_store, knowledge_graph, entity_extractor,
                 alpha=0.6,
                 vector_k=15,
                 graph_hops=2,
                 relevance_floor=0.25,
                 max_tokens=3000,
                 use_reranker=False,
                 reranker_model='cross-encoder/ms-marco-MiniLM-L-6-v2'):
        self.vector_store = vector_store
        self.knowledge_graph = knowledge_graph
        self.entity_extractor = entity_extractor
        self.alpha = alpha
        self.vector_k = vector_k
        self.graph_hops = graph_hops
        self.relevance_floor = relevance_floor
        self.max_tokens = max_tokens
        self.use_reranker = use_reranker
        self.reranker = None
        if self.use_reranker and CrossEncoder is not None:
            self.reranker = CrossEncoder(reranker_model)
        
        self.encoding = tiktoken.get_encoding("cl100k_base")

    def retrieve(self, query: str) -> RetrievalResult:
        # 1. Vector search
        vector_results = []
        if self.vector_store:
            vector_results = self.vector_store.query(query, self.vector_k)
        
        # 2. Extract entities from query
        entities = []
        if self.entity_extractor:
            entities, _ = self.entity_extractor.extract(query, 'query', 'query')
            
        # 3. Match entity names in graph
        # Entity objects have a .name attribute; extract the string names
        entity_names = []
        for e in entities:
            name = e.name if hasattr(e, 'name') else str(e)
            if name:
                entity_names.append(name)
        matched = entity_names
        
        # 4. Get graph chunks
        graph_chunks = {}
        if self.knowledge_graph and matched:
            graph_chunks = self.knowledge_graph.get_all_chunks_for_entities(matched, hops=self.graph_hops)
            
        # 5. Fuse results
        graph_score_map = {'direct': 1.0, 'hop_1': 0.7, 'hop_2': 0.4}
        
        # Helper to get text for graph-only chunks
        chunk_dict = {}
        for res in vector_results:
            chunk_dict[res['chunk_id']] = {
                'text': res['text'],
                'vector_score': res['score'],
                'metadata': res.get('metadata', {})
            }
            
        graph_only_ids = [cid for cid in graph_chunks.keys() if cid not in chunk_dict]
        
        if graph_only_ids and self.vector_store:
            try:
                # Direct chromadb query for texts
                res = self.vector_store.collection.get(ids=graph_only_ids)
                if res and res.get('documents'):
                    for i, cid in enumerate(graph_only_ids):
                        if i < len(res['documents']):
                            chunk_dict[cid] = {
                                'text': res['documents'][i],
                                'vector_score': 0.0,
                                'metadata': res['metadatas'][i] if res.get('metadatas') else {}
                            }
            except Exception as e:
                logger.warning(f"Failed to fetch text for graph-only chunks: {e}")
                
        fused_results = []
        for cid, data in chunk_dict.items():
            vector_score = data.get('vector_score', 0.0)
            graph_data = graph_chunks.get(cid, {})
            
            graph_source = graph_data.get('source')
            graph_score = graph_score_map.get(graph_source, 0.0)
            
            in_vector = vector_score > 0
            in_graph = graph_score > 0
            
            if in_vector and in_graph:
                combined_score = max(self.alpha * vector_score + (1 - self.alpha) * graph_score, vector_score, graph_score)
                source_tag = 'both'
            elif in_vector:
                combined_score = self.alpha * vector_score
                source_tag = 'direct_vector'
            elif in_graph:
                combined_score = (1 - self.alpha) * graph_score
                source_tag = f'graph_{graph_source}'
            else:
                continue
                
            if combined_score >= self.relevance_floor:
                fused_results.append({
                    'chunk_id': cid,
                    'text': data.get('text', ''),
                    'score': combined_score,
                    'source_tag': source_tag,
                    'metadata': data.get('metadata', {})
                })
                
        # 7. Sort by combined_score descending
        fused_results.sort(key=lambda x: x['score'], reverse=True)
        
        # 8. Token budget selection
        selected_chunks = []
        current_tokens = 0
        for chunk in fused_results:
            tokens = len(self.encoding.encode(chunk['text']))
            if current_tokens + tokens > self.max_tokens:
                break
            selected_chunks.append(chunk)
            current_tokens += tokens
            
        # 9. Reranking
        reranked = False
        if self.use_reranker and self.reranker and selected_chunks:
            pairs = [[query, chunk['text']] for chunk in selected_chunks]
            scores = self.reranker.predict(pairs)
            for i, chunk in enumerate(selected_chunks):
                chunk['score'] = float(scores[i])
            selected_chunks.sort(key=lambda x: x['score'], reverse=True)
            reranked = True
            
        return RetrievalResult(
            chunks=selected_chunks,
            query=query,
            vector_results_count=len(vector_results),
            graph_results_count=len(graph_chunks),
            fused_count=len(fused_results),
            reranked=reranked
        )


if __name__ == "__main__":
    # Smoke test
    class MockVectorStore:
        def query(self, q, k):
            return [{'chunk_id': 'c1', 'text': 'Vector text 1', 'score': 0.9, 'metadata': {}},
                    {'chunk_id': 'c2', 'text': 'Vector text 2', 'score': 0.7, 'metadata': {}}]
            
        class MockCollection:
            def get(self, ids):
                return {'documents': ['Graph text 1'], 'metadatas': [{}]}
        
        collection = MockCollection()
        
    class MockKG:
        def get_all_chunks_for_entities(self, entities, hops):
            return {'c2': {'source': 'hop_1', 'entity': 'test'}, 'c3': {'source': 'direct', 'entity': 'test'}}
            
    class MockExtractor:
        def extract(self, t, c, d):
            return (['entity1'], [])
            
    retriever = HybridRetriever(MockVectorStore(), MockKG(), MockExtractor())
    res = retriever.retrieve("test query")
    print(f"Retrieved {len(res.chunks)} chunks.")
    for c in res.chunks:
        print(c)
