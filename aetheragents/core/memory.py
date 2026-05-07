import chromadb
from typing import List, Dict, Optional
from datetime import datetime

class MemoryManager:
    def __init__(self, agent_name: str):
        self.client = chromadb.PersistentClient(path="./chroma_db")
        self.collection = self.client.get_or_create_collection(f"mem_{agent_name}")
        self.short_term: List[Dict] = []

    def add_message(self, role: str, content: str, **kwargs):
        entry = {"role": role, "content": content, **kwargs, "timestamp": str(datetime.now())}
        self.short_term.append(entry)
        # Periodically add to vector DB
        if len(self.short_term) % 3 == 0:
            self.collection.add(
                documents=[content],
                metadatas=[{"role": role}],
                ids=[f"{len(self.short_term)}"]
            )

    def get_recent_context(self, k: int = 15) -> List[Dict]:
        return self.short_term[-k:]

    def search(self, query: str, n_results: int = 5):
        return self.collection.query(query_texts=[query], n_results=n_results)