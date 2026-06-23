from __future__ import annotations

import json
import re
import uuid
from pathlib import Path


class ChromaVectorStore:
    """Chroma adapter with a dependency-free lexical fallback for the MVP."""

    def __init__(self, persist_dir: Path, prefer_chroma: bool = True):
        self.persist_dir = Path(persist_dir)
        self.persist_dir.mkdir(parents=True, exist_ok=True)
        self.fallback_path = self.persist_dir / "rules.json"
        self.collection = None
        if prefer_chroma:
            try:
                import chromadb

                client = chromadb.PersistentClient(path=str(self.persist_dir / "chroma"))
                self.collection = client.get_or_create_collection("automotive_body_rules")
            except (ImportError, RuntimeError, ValueError):
                self.collection = None

    def add(self, title: str, text: str, tags: list[str]) -> str:
        rule_id = uuid.uuid4().hex
        if self.collection is not None:
            self.collection.add(
                ids=[rule_id], documents=[text],
                metadatas=[{"title": title, "tags": ",".join(tags)}],
            )
        else:
            rules = self._load_fallback()
            rules.append({"id": rule_id, "title": title, "text": text, "tags": tags})
            self.fallback_path.write_text(
                json.dumps(rules, ensure_ascii=False, indent=2), encoding="utf-8"
            )
        return rule_id

    def list(self) -> list[dict]:
        if self.collection is not None:
            result = self.collection.get(include=["documents", "metadatas"])
            rules = []
            for index, rule_id in enumerate(result.get("ids", [])):
                metadata = result["metadatas"][index] or {}
                rules.append(
                    {
                        "id": rule_id,
                        "title": metadata.get("title", "Regle"),
                        "text": result["documents"][index],
                        "tags": [tag for tag in metadata.get("tags", "").split(",") if tag],
                    }
                )
            return rules
        return self._load_fallback()

    def search(self, query: str, limit: int = 5) -> list[dict]:
        if self.collection is not None and self.collection.count() > 0:
            result = self.collection.query(
                query_texts=[query], n_results=min(limit, self.collection.count()),
                include=["documents", "metadatas", "distances"],
            )
            rules = []
            for index, rule_id in enumerate(result["ids"][0]):
                metadata = result["metadatas"][0][index] or {}
                rules.append(
                    {
                        "id": rule_id,
                        "title": metadata.get("title", "Regle"),
                        "text": result["documents"][0][index],
                        "tags": [tag for tag in metadata.get("tags", "").split(",") if tag],
                        "score": 1.0 - float(result["distances"][0][index]),
                    }
                )
            return rules
        query_tokens = self._tokens(query)
        scored = []
        for rule in self.list():
            tokens = self._tokens(
                f"{rule['title']} {rule['text']} {' '.join(rule.get('tags', []))}"
            )
            score = len(tokens & query_tokens) / max(1, len(query_tokens))
            scored.append((score, rule))
        scored.sort(key=lambda pair: pair[0], reverse=True)
        return [{**rule, "score": score} for score, rule in scored[:limit]]

    def count(self) -> int:
        return self.collection.count() if self.collection is not None else len(self._load_fallback())

    def _load_fallback(self) -> list[dict]:
        if not self.fallback_path.exists():
            return []
        try:
            return json.loads(self.fallback_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return []

    @staticmethod
    def _tokens(text: str) -> set[str]:
        return {word for word in re.findall(r"[a-z0-9_]+", text.lower()) if len(word) > 2}

