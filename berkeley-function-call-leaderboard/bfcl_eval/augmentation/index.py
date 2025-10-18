from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np


@dataclass
class IndexMeta:
    embedding_model: str
    dim: int


class SimpleFaissIndex:
    def __init__(self, index_dir: Path, embedding_model: str) -> None:
        self.index_dir = index_dir
        self.embedding_model = embedding_model
        self.index_path = index_dir / "faiss.index"
        self.meta_path = index_dir / "meta.json"
        self.keys_path = index_dir / "keys.jsonl"

        # Lazy import to avoid mandatory dependency if users don't run augment
        import faiss  # type: ignore

        self.faiss = faiss
        self.index = None
        self.keys: List[dict] = []

    @staticmethod
    def _normalize(x: np.ndarray) -> np.ndarray:
        norms = np.linalg.norm(x, axis=1, keepdims=True) + 1e-12
        return x / norms

    def build(self, vectors: np.ndarray) -> None:
        vectors = self._normalize(vectors.astype(np.float32))
        dim = vectors.shape[1]
        self.index = self.faiss.IndexFlatIP(dim)
        self.index.add(vectors)
        meta = IndexMeta(embedding_model=self.embedding_model, dim=dim)
        self.index_dir.mkdir(parents=True, exist_ok=True)
        self.faiss.write_index(self.index, str(self.index_path))
        with open(self.meta_path, "w") as f:
            json.dump(meta.__dict__, f)

    def load(self) -> None:
        self.index = self.faiss.read_index(str(self.index_path))
        # load keys
        self.keys = []
        if self.keys_path.exists():
            with open(self.keys_path, "r") as f:
                for line in f:
                    self.keys.append(json.loads(line))

    def save_keys(self, keys: List[dict]) -> None:
        self.keys = keys
        self.index_dir.mkdir(parents=True, exist_ok=True)
        with open(self.keys_path, "w", encoding="utf-8") as f:
            for k in keys:
                f.write(json.dumps(k, ensure_ascii=False) + "\n")

    def search(
        self, query_vectors: np.ndarray, k: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        assert self.index is not None
        query_vectors = self._normalize(query_vectors.astype(np.float32))
        D, I = self.index.search(query_vectors, k)
        return D, I


def encode_texts(texts: List[str], model_name: str) -> np.ndarray:
    # Lazy import to avoid global dependency
    from sentence_transformers import SentenceTransformer  # type: ignore

    model = SentenceTransformer(model_name)
    emb = model.encode(texts, normalize_embeddings=False, convert_to_numpy=True)
    return emb.astype(np.float32)


def scope_filter_keys(
    keys: List[dict], target_subcategory: str, scope: str, k: int
) -> List[int]:
    if scope == "subcategory":
        indices = [
            i
            for i, krec in enumerate(keys)
            if krec.get("subcategory") == target_subcategory
        ]
        if len(indices) >= k:
            return indices
        # fallback: allow all agentic
        return list(range(len(keys)))
    else:
        return list(range(len(keys)))
