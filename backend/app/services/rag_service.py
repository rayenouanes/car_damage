from __future__ import annotations

from backend.app.adapters.vector_store_chroma import ChromaVectorStore


INITIAL_RULES = (
    "Un reflet change selon l'angle et la lumiere.",
    "Une vraie rayure reste generalement visible sur plusieurs frames.",
    "Une bosse legere se voit souvent par une deformation du reflet, pas seulement par une trace.",
    "La salete peut ressembler a un impact mais n'a pas forcement de rupture visuelle nette.",
    "Sur carrosserie noire brillante, les reflets blancs longs sont souvent confondus avec des rayures.",
    "Ne pas annoter toute la portiere : annoter uniquement la zone du defaut.",
)


class RAGService:
    def __init__(self, store: ChromaVectorStore):
        self.store = store

    def seed_initial_rules(self) -> None:
        if self.store.count() == 0:
            for index, text in enumerate(INITIAL_RULES, start=1):
                self.add_rule(f"Regle metier {index}", text, ["carrosserie"])

    def add_rule(self, title: str, text: str, tags: list[str]) -> str:
        return self.store.add(title.strip(), text.strip(), tags)

    def list_rules(self) -> list[dict]:
        return self.store.list()

    def search(self, query: str, limit: int = 5) -> list[dict]:
        return self.store.search(query, limit)

    def relevant_text(self, prediction: dict, limit: int = 5) -> str:
        query = (
            f"{prediction.get('class_name', '')} {' '.join(prediction.get('active_learning_reasons', []))} "
            "reflet ombre salete bbox carrosserie"
        )
        rules = self.search(query, limit)
        return "\n".join(f"- {rule['text']}" for rule in rules)

