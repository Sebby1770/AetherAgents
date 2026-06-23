"""Agent memory: a rolling short-term window plus a pluggable long-term store.

The original implementation hard-required ChromaDB at import time, which made
the whole package unimportable without it. Memory now defaults to a dependency
-free :class:`InMemoryVectorStore` (bag-of-words cosine similarity) and treats
Chroma as an optional backend you can opt into.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from datetime import datetime, timezone
from typing import Any, Protocol, runtime_checkable

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


@runtime_checkable
class VectorStore(Protocol):
    """Minimal contract for a long-term memory backend."""

    def add(self, id: str, text: str, metadata: dict[str, Any] | None = None) -> None: ...

    def query(self, text: str, k: int = 5) -> list[dict[str, Any]]: ...

    def count(self) -> int: ...


class InMemoryVectorStore:
    """A tiny, dependency-free semantic store.

    Uses term-frequency cosine similarity. Not a substitute for real embeddings
    at scale, but deterministic, fast and perfect for tests and small agents.
    """

    def __init__(self) -> None:
        self._docs: list[dict[str, Any]] = []
        self._vectors: list[Counter] = []

    def add(self, id: str, text: str, metadata: dict[str, Any] | None = None) -> None:
        self._docs.append({"id": id, "document": text, "metadata": metadata or {}})
        self._vectors.append(Counter(_tokenize(text)))

    def query(self, text: str, k: int = 5) -> list[dict[str, Any]]:
        q = Counter(_tokenize(text))
        if not q:
            return []
        scored: list[dict[str, Any]] = []
        for doc, vec in zip(self._docs, self._vectors, strict=False):
            score = _cosine(q, vec)
            if score > 0:
                scored.append({**doc, "score": score})
        scored.sort(key=lambda d: d["score"], reverse=True)
        return scored[:k]

    def count(self) -> int:
        return len(self._docs)


def _cosine(a: Counter, b: Counter) -> float:
    if not a or not b:
        return 0.0
    common = set(a) & set(b)
    dot = sum(a[t] * b[t] for t in common)
    if dot == 0:
        return 0.0
    na = math.sqrt(sum(v * v for v in a.values()))
    nb = math.sqrt(sum(v * v for v in b.values()))
    return dot / (na * nb)


class ChromaVectorStore:
    """A long-term store backed by ChromaDB (optional dependency)."""

    def __init__(self, collection: str, path: str = "./chroma_db") -> None:
        try:
            import chromadb
        except ImportError as exc:  # pragma: no cover - depends on env
            raise ImportError(
                "chromadb is not installed. Install it with: pip install 'aetheragents[chroma]'"
            ) from exc
        self._client = chromadb.PersistentClient(path=path)
        self._collection = self._client.get_or_create_collection(collection)

    def add(self, id: str, text: str, metadata: dict[str, Any] | None = None) -> None:
        self._collection.add(documents=[text], metadatas=[metadata or {}], ids=[id])

    def query(self, text: str, k: int = 5) -> list[dict[str, Any]]:
        res = self._collection.query(query_texts=[text], n_results=k)
        out: list[dict[str, Any]] = []
        ids = (res.get("ids") or [[]])[0]
        docs = (res.get("documents") or [[]])[0]
        metas = (res.get("metadatas") or [[]])[0]
        dists = (res.get("distances") or [[]])[0]
        for i, doc in enumerate(docs):
            out.append(
                {
                    "id": ids[i] if i < len(ids) else str(i),
                    "document": doc,
                    "metadata": metas[i] if i < len(metas) else {},
                    "score": 1.0 - dists[i] if i < len(dists) else None,
                }
            )
        return out

    def count(self) -> int:
        return self._collection.count()


class MemoryManager:
    """Short-term conversation window + long-term vector recall."""

    def __init__(
        self,
        agent_name: str,
        *,
        vector_store: VectorStore | None = None,
        window: int = 20,
        persist_every: int = 1,
    ) -> None:
        self.agent_name = agent_name
        self.window = window
        self.persist_every = max(1, persist_every)
        self.store: VectorStore = vector_store or InMemoryVectorStore()
        self.short_term: list[dict[str, Any]] = []
        self._seq = 0

    def add_message(self, role: str, content: str, **metadata: Any) -> None:
        entry = {
            "role": role,
            "content": content,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            **metadata,
        }
        self.short_term.append(entry)
        self._seq += 1
        if content and self._seq % self.persist_every == 0:
            self.store.add(
                id=f"{self.agent_name}-{self._seq}",
                text=content,
                metadata={"role": role, **metadata},
            )

    def get_recent_context(self, k: int = 15) -> list[dict[str, Any]]:
        return self.short_term[-k:]

    def search(self, query: str, n_results: int = 5) -> list[dict[str, Any]]:
        return self.store.query(query, k=n_results)

    # Friendlier aliases.
    def remember(self, content: str, **metadata: Any) -> None:
        self.add_message("memory", content, **metadata)

    def recall(self, query: str, k: int = 5) -> list[dict[str, Any]]:
        return self.search(query, n_results=k)

    def clear(self) -> None:
        self.short_term.clear()
