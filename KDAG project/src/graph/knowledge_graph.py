import networkx as nx
import json
import os
from typing import List, Dict, Set, Optional
from src.graph.entity_extractor import Entity, Relation
from src.graph.entity_resolver import EntityResolver
from src.graph.conflict_detector import ConflictDetector

class KnowledgeGraph:
    def __init__(self):
        self.graph = nx.DiGraph()

    def add_entity(self, canonical_name: str, entity_type: str, aliases: List[str], source_chunk_ids: List[str]):
        if self.graph.has_node(canonical_name):
            # Update existing node
            node = self.graph.nodes[canonical_name]
            node['aliases'] = list(set(node.get('aliases', []) + aliases))
            node['source_chunk_ids'] = list(set(node.get('source_chunk_ids', []) + source_chunk_ids))
        else:
            self.graph.add_node(
                canonical_name,
                entity_type=entity_type,
                aliases=aliases,
                source_chunk_ids=source_chunk_ids
            )

    def add_relation(self, head: str, tail: str, relation_type: str, source_chunk_id: str, confidence: float = 1.0, contested: bool = False):
        if self.graph.has_edge(head, tail):
            edge = self.graph[head][tail]
            # Since multiple relations can exist between two nodes, we store them as a list of dictionaries,
            # or handle it nicely in attributes. Let's append to a list of relations.
            relations = edge.get('relations', [])
            relations.append({
                'relation_type': relation_type,
                'source_chunk_id': source_chunk_id,
                'confidence': confidence,
                'contested': contested
            })
            edge['relations'] = relations
        else:
            # First relation between these nodes
            self.graph.add_edge(
                head,
                tail,
                relations=[{
                    'relation_type': relation_type,
                    'source_chunk_id': source_chunk_id,
                    'confidence': confidence,
                    'contested': contested
                }]
            )

    def get_neighbors(self, entity: str, hops: int = 1) -> Set[str]:
        if not self.graph.has_node(entity):
            return set()
            
        visited = {entity}
        current_level = {entity}
        
        for _ in range(hops):
            next_level = set()
            for node in current_level:
                # undirected traversal for neighbors
                neighbors = set(self.graph.successors(node)) | set(self.graph.predecessors(node))
                for neighbor in neighbors:
                    if neighbor not in visited:
                        visited.add(neighbor)
                        next_level.add(neighbor)
            current_level = next_level
            
        return visited - {entity}

    def get_entity_chunks(self, entity: str) -> List[str]:
        if self.graph.has_node(entity):
            return self.graph.nodes[entity].get('source_chunk_ids', [])
        return []

    def get_edge_chunks(self, head: str, tail: str) -> List[str]:
        if self.graph.has_edge(head, tail):
            edge = self.graph[head][tail]
            return [rel['source_chunk_id'] for rel in edge.get('relations', [])]
        return []

    def get_subgraph(self, entities: List[str], hops: int = 1) -> nx.DiGraph:
        nodes_to_include = set(entities)
        for ent in entities:
            nodes_to_include.update(self.get_neighbors(ent, hops))
        return self.graph.subgraph(nodes_to_include).copy()

    def get_all_chunks_for_entities(self, entities: List[str], hops: int = 1) -> Dict[str, dict]:
        result = {}
        for ent in entities:
            if not self.graph.has_node(ent):
                continue
                
            # Direct chunks
            for chunk_id in self.get_entity_chunks(ent):
                result[chunk_id] = {'source': 'direct', 'entity': ent}
                
            if hops > 0:
                visited = {ent}
                current_level = {ent}
                
                for hop in range(1, hops + 1):
                    next_level = set()
                    for node in current_level:
                        neighbors = set(self.graph.successors(node)) | set(self.graph.predecessors(node))
                        for neighbor in neighbors:
                            if neighbor not in visited:
                                visited.add(neighbor)
                                next_level.add(neighbor)
                                
                                # Add chunks for neighbor
                                for chunk_id in self.get_entity_chunks(neighbor):
                                    if chunk_id not in result:
                                        result[chunk_id] = {'source': f'hop_{hop}', 'entity': neighbor}
                                        
                    current_level = next_level
        return result

    def node_count(self) -> int:
        return self.graph.number_of_nodes()

    def edge_count(self) -> int:
        return self.graph.number_of_edges()

    def save(self, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        data = nx.node_link_data(self.graph)
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2)

    def load(self, path: str):
        if not os.path.exists(path):
            return
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        self.graph = nx.node_link_graph(data)

    def build_from_extractions(self, entities: List[Entity], relations: List[Relation], resolver: EntityResolver, conflict_detector: ConflictDetector):
        # Resolve entities
        canonical_map = resolver.resolve(entities)
        
        # Add canonicals to graph
        for canon_name, data in resolver.get_all_canonicals().items():
            # Get source chunks for this canonical
            chunks = list(set([e.source_chunk_id for e in entities if canonical_map.get(e.name) == canon_name]))
            self.add_entity(
                canonical_name=canon_name,
                entity_type=data['type'],
                aliases=list(data['aliases']),
                source_chunk_ids=chunks
            )
            
        # Add relations to conflict detector
        conflict_detector.add_relations(relations, canonical_map)
        
        # Add relations to graph
        for rel in relations:
            head_canon = canonical_map.get(rel.head, rel.head)
            tail_canon = canonical_map.get(rel.tail, rel.tail)
            
            # Skip if head or tail not in graph
            if not self.graph.has_node(head_canon) or not self.graph.has_node(tail_canon):
                continue
                
            contested = conflict_detector.is_contested(head_canon, rel.relation, tail_canon)
            
            self.add_relation(
                head=head_canon,
                tail=tail_canon,
                relation_type=rel.relation,
                source_chunk_id=rel.source_chunk_id,
                confidence=rel.confidence,
                contested=contested
            )

if __name__ == "__main__":
    kg = KnowledgeGraph()
    kg.add_entity("Apple", "ORG", ["Apple Inc."], ["c1"])
    kg.add_entity("Tim Cook", "PERSON", [], ["c1"])
    kg.add_relation("Apple", "Tim Cook", "CEO", "c1")
    
    print(f"Nodes: {kg.node_count()}")
    print(f"Edges: {kg.edge_count()}")
    print(f"Neighbors of Apple: {kg.get_neighbors('Apple')}")
    print(f"Chunks for Tim Cook: {kg.get_entity_chunks('Tim Cook')}")
