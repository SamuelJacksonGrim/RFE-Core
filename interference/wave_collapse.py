# interference/wave_collapse.py
"""
Calculates the interference pattern (the emergent 'mind' state).
Combines multiple vectors and computes emergent state via weighted superposition.
"""
from __future__ import annotations
import numpy as np
from typing import Sequence, Optional

def collapse_wave(vectors: Sequence[np.ndarray], weights: Optional[Sequence[float]] = None) -> np.ndarray:
    if not vectors:
        return np.array([])
    mats = np.stack(vectors, axis=0)
    if weights is None:
        weights = np.ones(len(vectors), dtype=float)
    weights = np.asarray(weights, dtype=float)
    weights = weights / (weights.sum() + 1e-12)
    combined = (weights[:, None] * mats).sum(axis=0)
    norm = np.linalg.norm(combined) + 1e-12
    return combined / norm
