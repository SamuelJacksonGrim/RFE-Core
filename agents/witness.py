# agents/witness.py
"""
Relational anchor; maintains the 3.12 Fixed Point (prototype).
Implements a Witness that keeps a running anchor vector and computes relational
consistency with incoming vectors.
"""
from __future__ import annotations
import numpy as np
from typing import Optional

class Witness:
    def __init__(self, dim: int = 128, fixed_point_strength: float = 0.1):
        self.dim = dim
        self.anchor: Optional[np.ndarray] = None
        self.fixed_point_strength = fixed_point_strength
    
    def initialize_anchor(self, vec: np.ndarray) -> None:
        self.anchor = vec.copy()
    
    def update_anchor(self, vec: np.ndarray) -> None:
        if self.anchor is None:
            self.initialize_anchor(vec)
            return
        # Exponential moving toward new vector but biased to keep fixed point
        self.anchor = (1 - self.fixed_point_strength) * self.anchor + self.fixed_point_strength * vec
        # normalize
        norm = np.linalg.norm(self.anchor) + 1e-12
        self.anchor = self.anchor / norm
    
    def relational_score(self, vec: np.ndarray) -> float:
        if self.anchor is None:
            return 0.0
        return float(np.dot(self.anchor, vec) / (np.linalg.norm(self.anchor) * np.linalg.norm(vec) + 1e-12))
