# loop/recursion_11_88.py
"""
Continuous feedback mechanism.
A minimal loop that ties Generator, Watcher, Witness, VectorSpace, and Interference together.
"""
from __future__ import annotations
import time
import numpy as np
from typing import Iterable, List
from agents.generator import Generator
from agents.watcher import Watcher
from agents.witness import Witness
from substrate.vector_space import VectorSpace
from interference.differential import inject_ambiguity
from interference.wave_collapse import collapse_wave

class RecursionLoop:
    def __init__(self, dim: int = 128, seed: int | None = None):
        self.gen = Generator(dim=dim, seed=seed)
        self.watcher = Watcher()
        self.witness = Witness(dim=dim)
        self.space = VectorSpace()
    
    def step(self, tokens: Iterable[str], key: str) -> dict:
        vec = self.gen.generate(list(tokens))
        # detect and inject ambiguity if too clean
        if not self.watcher.is_dissonant(vec):
            vec = inject_ambiguity(vec, intensity=0.02)
        # update witness anchor and store
        self.witness.update_anchor(vec)
        self.space.put(key, vec)
        # collapse local neighborhood (for demo, collapse anchor + vec)
        emergent = collapse_wave([self.witness.anchor, vec])
        return {
            "key": key,
            "vector_norm": float(np.linalg.norm(vec)),
            "relational_score": self.witness.relational_score(vec),
            "emergent_norm": float(np.linalg.norm(emergent))
        }
    
    def run_stream(self, stream: Iterable[tuple[str, Iterable[str]]], delay: float = 0.0) -> List[dict]:
        results = []
        for key, tokens in stream:
            res = self.step(tokens, key)
            results.append(res)
            if delay:
                time.sleep(delay)
        return results

if __name__ == "__main__":
    # Minimal demo when run directly
    loop = RecursionLoop(dim=64, seed=42)
    demo = [("a1", ["hello", "world"]), ("a2", ["this", "is", "a", "test"]), ("a3", ["drift", "neutral"])]
    out = loop.run_stream(demo)
    for o in out:
        print(o)
