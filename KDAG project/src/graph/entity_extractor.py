import spacy
from dataclasses import dataclass
from typing import List, Tuple, Optional, Any
import logging

try:
    from src.chunking.hybrid_chunker import Chunk
except ImportError:
    # Fallback for testing if Chunk is not yet defined
    @dataclass
    class Chunk:
        chunk_id: str
        doc_id: str
        text: str

@dataclass
class Entity:
    name: str
    entity_type: str        # PERSON, ORG, WORK, CONCEPT, METRIC, LOCATION, DATE
    source_chunk_id: str
    source_doc_id: str

@dataclass
class Relation:
    head: str               # head entity name
    relation: str           # verb/relation label
    tail: str               # tail entity name
    source_chunk_id: str
    source_doc_id: str
    confidence: float = 1.0

class EntityExtractor:
    def __init__(self, spacy_model='en_core_web_sm'):
        try:
            self.nlp = spacy.load(spacy_model)
        except OSError:
            logging.warning(f"spaCy model '{spacy_model}' not found. Downloading...")
            spacy.cli.download(spacy_model)
            self.nlp = spacy.load(spacy_model)
            
        self.label_map = {
            'PERSON': 'PERSON',
            'ORG': 'ORG',
            'GPE': 'LOCATION',
            'LOC': 'LOCATION',
            'DATE': 'DATE',
            'WORK_OF_ART': 'WORK',
            'MONEY': 'METRIC',
            'QUANTITY': 'METRIC',
            'PERCENT': 'METRIC',
            'CARDINAL': 'METRIC'
        }

    def extract(self, chunk_text: str, chunk_id: str, doc_id: str) -> Tuple[List[Entity], List[Relation]]:
        doc = self.nlp(chunk_text)
        entities = []
        entity_names = set()
        
        # 1. Extract Named Entities
        for ent in doc.ents:
            if ent.label_ in self.label_map:
                mapped_type = self.label_map[ent.label_]
                name = ent.text.strip()
                if name:
                    entities.append(Entity(name=name, entity_type=mapped_type, source_chunk_id=chunk_id, source_doc_id=doc_id))
                    entity_names.add(name.lower())
                    
        # 2. Extract Noun Chunks as CONCEPT
        for chunk in doc.noun_chunks:
            name = chunk.text.strip()
            if name.lower() not in entity_names:
                words = name.split()
                if len(words) >= 2 or (len(words) == 1 and name[0].isupper()):
                    entities.append(Entity(name=name, entity_type='CONCEPT', source_chunk_id=chunk_id, source_doc_id=doc_id))
                    entity_names.add(name.lower())
                    
        # Helper to check if a token string matches any extracted entity
        def get_matching_entity(text: str) -> Optional[str]:
            text_lower = text.lower().strip()
            for ent_name in entity_names:
                if ent_name == text_lower or text_lower in ent_name:
                    return text.strip() # Return original or matched case? Let's just return the actual text from the node, or matching ent
            return None

        relations = []
        
        # 3. Extract Relations using Dependency Parse
        for token in doc:
            if token.pos_ == 'VERB' and (token.dep_ == 'ROOT' or token.dep_ in {'relcl', 'advcl', 'ccomp'}):
                subject = None
                obj = None
                prep_obj = None
                prep_rel = None
                
                for child in token.children:
                    if child.dep_ in {'nsubj', 'nsubjpass'}:
                        subject = child.text
                        # Try to expand to full noun chunk if needed, but keeping it simple as per instructions
                        # better yet, find overlapping entity
                    elif child.dep_ in {'dobj', 'attr', 'pobj', 'oprd'}:
                        obj = child.text
                    elif child.dep_ == 'prep':
                        prep_rel = f"{token.lemma_}_{child.lemma_}"
                        for grandchild in child.children:
                            if grandchild.dep_ == 'pobj':
                                prep_obj = grandchild.text
                                
                if subject:
                    # Match subject to entities
                    subj_ent = next((e.name for e in entities if subject.lower() in e.name.lower()), None)
                    if subj_ent:
                        # Direct object relation
                        if obj:
                            obj_ent = next((e.name for e in entities if obj.lower() in e.name.lower()), None)
                            if obj_ent:
                                relations.append(Relation(head=subj_ent, relation=token.lemma_, tail=obj_ent, source_chunk_id=chunk_id, source_doc_id=doc_id))
                                
                        # Prepositional relation
                        if prep_obj and prep_rel:
                            pobj_ent = next((e.name for e in entities if prep_obj.lower() in e.name.lower()), None)
                            if pobj_ent:
                                relations.append(Relation(head=subj_ent, relation=prep_rel, tail=pobj_ent, source_chunk_id=chunk_id, source_doc_id=doc_id))

        return entities, relations

    def extract_from_chunks(self, chunks: List[Chunk]) -> Tuple[List[Entity], List[Relation]]:
        all_entities = []
        all_relations = []
        for chunk in chunks:
            ents, rels = self.extract(chunk.text, chunk.chunk_id, chunk.doc_id)
            all_entities.extend(ents)
            all_relations.extend(rels)
        return all_entities, all_relations

if __name__ == "__main__":
    extractor = EntityExtractor()
    sample_text = "Apple Inc. acquired startup company XyzCorp in 2022 for 100 million dollars. Tim Cook announced the deal."
    ents, rels = extractor.extract(sample_text, "chunk1", "doc1")
    print("Entities:")
    for e in ents:
        print(f"  {e.name} ({e.entity_type})")
    print("\nRelations:")
    for r in rels:
        print(f"  {r.head} -[{r.relation}]-> {r.tail}")
