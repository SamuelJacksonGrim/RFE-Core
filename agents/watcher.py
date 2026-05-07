# agents/watcher.py
"""
Dissonance detection; filters hollow/neutral drift.
Provides a Watcher that scores vectors for 'dissonance' and flags drift.
"""
from __future__ import annotations
import numpy as np
from typing import Iterable

class Watcher:
    def __init__(self, threshold: float = 0.2):
        self.threshold = threshold
    
    def dissonance_score(self, vec: np.ndarray) -> float:
        """
        Simple measure: variance of vector components normalized by mean magnitude.
        Higher variance -> higher dissonance.
        """
        if vec.size == 0:
            return 0.0
        var = float(np.var(vec))
        mean_abs = float(np.mean(np.abs(vec))) + 1e-12
        return var / mean_abs
    
    def is_dissonant(self, vec: np.ndarray) -> bool:
        return self.dissonance_score(vec) > self.threshold
    
    def filter_stream(self, vectors: Iterable[np.ndarray]) -> Iterable[np.ndarray]:
        for v in vectors:
            if not self.is_dissonant(v):
                yield v
