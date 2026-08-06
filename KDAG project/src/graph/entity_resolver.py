import json
import os
import numpy as np
from typing import List, Dict, Optional, Any
from sentence_transformers import SentenceTransformer
from src.graph.entity_extractor import Entity

class EntityResolver:
    def __init__(self, embedding_model_name='all-MiniLM-L6-v2', similarity_threshold=0.85):
        self.similarity_threshold = similarity_threshold
        self.model = SentenceTransformer(embedding_model_name)
        # mapping canonical_name -> {type, aliases: set, embedding: np.ndarray}
        self.canonical_registry: Dict[str, dict] = {}

    def _normalize(self, text: str) -> str:
        return text.strip().lower()

    def resolve(self, entities: List[Entity]) -> Dict[str, str]:
        resolved_map = {}
        
        for ent in entities:
            norm_name = self._normalize(ent.name)
            
            # 1. Exact match
            matched = False
            for canon_name, data in self.canonical_registry.items():
                if norm_name == self._normalize(canon_name) or norm_name in data['aliases']:
                    resolved_map[ent.name] = canon_name
                    matched = True
                    break
                    
            if matched:
                continue
                
            # 2. Embedding match
            ent_emb = self.model.encode(ent.name)
            best_match = None
            best_sim = -1.0
            
            for canon_name, data in self.canonical_registry.items():
                canon_emb = data['embedding']
                sim = np.dot(ent_emb, canon_emb) / (np.linalg.norm(ent_emb) * np.linalg.norm(canon_emb))
                
                if sim > best_sim:
                    best_sim = sim
                    best_match = canon_name
                    
            if best_match and best_sim >= self.similarity_threshold:
                match_data = self.canonical_registry[best_match]
                # Type compatibility check
                if match_data['type'] == ent.entity_type or ent.entity_type == 'CONCEPT' or match_data['type'] == 'CONCEPT':
                    match_data['aliases'].add(norm_name)
                    # Update embedding as mean
                    match_data['embedding'] = (match_data['embedding'] + ent_emb) / 2.0
                    resolved_map[ent.name] = best_match
                    matched = True
                    
            if matched:
                continue
                
            # 3. New canonical
            self.canonical_registry[ent.name] = {
                'type': ent.entity_type,
                'aliases': {norm_name},
                'embedding': ent_emb
            }
            resolved_map[ent.name] = ent.name

        return resolved_map

    def get_canonical(self, name: str) -> Optional[str]:
        norm_name = self._normalize(name)
        for canon_name, data in self.canonical_registry.items():
            if norm_name == self._normalize(canon_name) or norm_name in data['aliases']:
                return canon_name
        return None

    def get_all_canonicals(self) -> Dict[str, dict]:
        return self.canonical_registry

    def save(self, path: str):
        # Convert ndarray and set for JSON
        serializable_reg = {}
        for k, v in self.canonical_registry.items():
            serializable_reg[k] = {
                'type': v['type'],
                'aliases': list(v['aliases']),
                'embedding': v['embedding'].tolist()
            }
        
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(serializable_reg, f, indent=2)

    def load(self, path: str):
        if not os.path.exists(path):
            return
            
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        self.canonical_registry = {}
        for k, v in data.items():
            self.canonical_registry[k] = {
                'type': v['type'],
                'aliases': set(v['aliases']),
                'embedding': np.array(v['embedding'])
            }

if __name__ == "__main__":
    resolver = EntityResolver()
    entities = [
        Entity("Apple", "ORG", "c1", "d1"),
        Entity("Apple Inc.", "ORG", "c2", "d2"),
        Entity("Apple Corporation", "ORG", "c3", "d3"),
        Entity("Banana", "CONCEPT", "c4", "d4")
    ]
    res_map = resolver.resolve(entities)
    print("Resolution Map:")
    for k, v in res_map.items():
        print(f"  {k} -> {v}")
