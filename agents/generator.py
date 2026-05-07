# agents/generator.py
"""
High-density pattern generation; seeks E_8 symmetries (prototype).
This module exposes a Generator class that produces high-dimensional
vectors from token-like inputs using configurable cadence rules.
"""
from __future__ import annotations
import numpy as np
from typing import Sequence, Dict

class Generator:
    def __init__(self, dim: int = 128, seed: int | None = None, rhythm: Dict | None = None):
        self.dim = dim
        self.rng = np.random.default_rng(seed)
        self.rhythm = rhythm or {}
    
    def _rhythm_scale(self, token: str) -> float:
        base = self.rhythm.get("base_scale", 1.0)
        token_hash = sum(ord(c) for c in token) % 997
        return base * (1.0 + (token_hash / 997.0) * 0.5)
    
    def generate(self, tokens: Sequence[str]) -> np.ndarray:
        """
        Generate a single vector representing the sequence of tokens.
        """
        if not tokens:
            return np.zeros(self.dim, dtype=float)
        vec = np.zeros(self.dim, dtype=float)
        for i, t in enumerate(tokens):
            scale = self._rhythm_scale(t)
            noise = self.rng.normal(loc=0.0, scale=0.1, size=self.dim)
            pattern = np.sin((i + 1) * np.linspace(0, np.pi * scale, self.dim))
            vec += pattern + noise
        # normalize
        norm = np.linalg.norm(vec)
        return vec / (norm + 1e-12)
