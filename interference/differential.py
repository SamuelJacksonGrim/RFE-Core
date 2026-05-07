# interference/differential.py
"""
Injects ambiguity/creative tension when vectors align too cleanly.
Provides a function to perturb vectors based on alignment metrics.
"""
from __future__ import annotations
import numpy as np

def inject_ambiguity(vec: np.ndarray, intensity: float = 0.05, rng: np.random.Generator | None = None) -> np.ndarray:
    rng = rng or np.random.default_rng()
    noise = rng.normal(scale=intensity, size=vec.shape)
    perturbed = vec + noise
    norm = np.linalg.norm(perturbed) + 1e-12
    return perturbed / norm
