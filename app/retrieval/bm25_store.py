import os
import json
from rank_bm25 import BM25Okapi
from langchain_core.documents import Document

CHUNK_DIR = "data/chunks"


class BM25Store:
    def __init__(self):
        self.documents = self._load_chunks()
        self.corpus = [doc.page_content.split() for doc in self.documents]
        self.bm25 = BM25Okapi(self.corpus)
        print(f"[BM25] Loaded {len(self.documents)} chunks")

    def _load_chunks(self):
        docs = []
        if not os.path.exists(CHUNK_DIR):
            print("[BM25] No chunk directory found")
            return docs

        for filename in sorted(os.listdir(CHUNK_DIR)):
            if not filename.endswith(".json"):
                continue

            with open(os.path.join(CHUNK_DIR, filename), "r", encoding="utf-8") as f:
                data = json.load(f)
                docs.append(Document(
                    page_content=data["page_content"],
                    metadata=data.get("metadata", {})
                ))

        return docs

    def search(self, query: str, k: int = 10):
        tokens = query.split()
        scores = self.bm25.get_scores(tokens)

        ranked = sorted(
            zip(self.documents, scores),
            key=lambda x: x[1],
            reverse=True
        )

        return [doc for doc, _ in ranked[:k]]


bm25 = BM25Store()
