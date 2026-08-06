from dataclasses import dataclass
from typing import List, Tuple, Dict
from src.graph.entity_extractor import Entity, Relation

@dataclass
class ContestedFact:
    entity_pair: Tuple[str, str]    # (head_canonical, tail_canonical)
    relation: str
    assertions: List[dict]          # [{value, source_chunk_id, source_doc_id}]

class ConflictDetector:
    def __init__(self):
        # key = (head_canonical, relation) -> list of {tail_canonical, chunk_id, doc_id}
        self.fact_store: Dict[Tuple[str, str], List[dict]] = {}

    def add_relations(self, relations: List[Relation], canonical_map: Dict[str, str]):
        for rel in relations:
            head_canon = canonical_map.get(rel.head, rel.head)
            tail_canon = canonical_map.get(rel.tail, rel.tail)
            
            key = (head_canon, rel.relation)
            if key not in self.fact_store:
                self.fact_store[key] = []
                
            self.fact_store[key].append({
                'value': tail_canon,
                'source_chunk_id': rel.source_chunk_id,
                'source_doc_id': rel.source_doc_id
            })

    def detect_conflicts(self) -> List[ContestedFact]:
        conflicts = []
        for (head, relation), assertions in self.fact_store.items():
            # Get unique tail values
            unique_tails = set(a['value'] for a in assertions)
            if len(unique_tails) > 1:
                # Group assertions by tail
                for tail in unique_tails:
                    tail_assertions = [a for a in assertions if a['value'] == tail]
                    # Create ContestedFact for each variant? Or one ContestedFact with all assertions?
                    # The prompt says: "if there are 2+ distinct tail values, create a ContestedFact"
                    # But entity_pair is (head_canonical, tail_canonical). So we create one per tail variant.
                    conflicts.append(ContestedFact(
                        entity_pair=(head, tail),
                        relation=relation,
                        assertions=tail_assertions
                    ))
        return conflicts

    def is_contested(self, head: str, relation: str, tail: str) -> bool:
        key = (head, relation)
        if key not in self.fact_store:
            return False
        unique_tails = set(a['value'] for a in self.fact_store[key])
        return len(unique_tails) > 1

    def get_all_contested(self) -> List[ContestedFact]:
        return self.detect_conflicts()

if __name__ == "__main__":
    detector = ConflictDetector()
    rels = [
        Relation("Apple", "CEO", "Tim Cook", "c1", "d1"),
        Relation("Apple", "CEO", "Steve Jobs", "c2", "d2")
    ]
    canonical_map = {"Apple": "Apple", "Tim Cook": "Tim Cook", "Steve Jobs": "Steve Jobs"}
    
    detector.add_relations(rels, canonical_map)
    conflicts = detector.detect_conflicts()
    print(f"Detected {len(conflicts)} conflicts:")
    for c in conflicts:
        print(f"  {c.entity_pair[0]} -[{c.relation}]-> {c.entity_pair[1]} (Assertions: {len(c.assertions)})")
