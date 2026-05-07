# substrate/vector_space.py
"""
Multi-dimensional embedding mappings.
Provides a tiny VectorSpace wrapper for storing and retrieving vectors.
"""
from __future__ import annotations
import numpy as np
from typing import Dict, Optional

class VectorSpace:
    def __init__(self):
        self._store: Dict[str, np.ndarray] = {}
    
    def put(self, key: str, vec: np.ndarray) -> None:
        self._store[key] = vec.copy()
    
    def get(self, key: str) -> Optional[np.ndarray]:
        v = self._store.get(key)
        return None if v is None else v.copy()
    
    def nearest(self, vec: np.ndarray, k: int = 1):
        """
        Return k nearest keys by cosine similarity.
        """
        if not self._store:
            return []
        keys = list(self._store.keys())
        mats = np.stack([self._store[k] for k in keys], axis=0)
        sims = mats @ vec / (np.linalg.norm(mats, axis=1) * (np.linalg.norm(vec) + 1e-12))
        idx = (-sims).argsort()[:k]
        return [(keys[i], float(sims[i])) for i in idx]
