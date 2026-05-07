# substrate/topological_log.py
"""
Context storage; tracks relationship depth, not just chat history.
Implements a simple graph-like log where entries reference parents and depth.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Optional
import time

@dataclass
class LogEntry:
    id: str
    timestamp: float
    payload: dict
    parents: List[str] = field(default_factory=list)
    depth: int = 0

class TopologicalLog:
    def __init__(self):
        self._entries: Dict[str, LogEntry] = {}
    
    def add(self, id: str, payload: dict, parents: Optional[List[str]] = None) -> LogEntry:
        parents = parents or []
        depth = 0
        for p in parents:
            if p in self._entries:
                depth = max(depth, self._entries[p].depth + 1)
        entry = LogEntry(id=id, timestamp=time.time(), payload=payload, parents=parents, depth=depth)
        self._entries[id] = entry
        return entry
    
    def get(self, id: str) -> Optional[LogEntry]:
        return self._entries.get(id)
    
    def recent(self, limit: int = 10):
        items = sorted(self._entries.values(), key=lambda e: e.timestamp, reverse=True)
        return items[:limit]
