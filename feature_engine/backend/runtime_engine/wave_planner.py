"""Section 1: Deterministic wave-based planning using Kahn's algorithm.

Usage example:
    from backend.runtime_engine.config import FEATURES
    from backend.runtime_engine.wave_planner import collect_transitive_features, build_waves

    required = {"F_EMBED", "F_USER_HIST"}
    feature_set = collect_transitive_features(required, FEATURES)
    waves = build_waves(feature_set, FEATURES)
    # Example: [['F_RAW_TEXT', 'F_USER_ID'], ['F_TOKENS', 'F_USER_HIST'], ['F_EMBED']]
"""

from __future__ import annotations

from collections import defaultdict, deque
from typing import Dict, List, Set


def collect_transitive_features(required_features: Set[str], features: Dict[str, Dict[str, object]]) -> Set[str]:
    """Return the minimal transitive closure for required features."""
    needed: Set[str] = set()

    def dfs(fid: str) -> None:
        if fid in needed:
            return
        if fid not in features:
            raise KeyError(f"Unknown feature: {fid}")
        needed.add(fid)
        deps = features[fid].get("deps", [])
        for dep in deps:
            if not isinstance(dep, str):
                raise TypeError(f"Dependency names must be strings. Invalid dependency in {fid}: {dep!r}")
            dfs(dep)

    for feature_id in sorted(required_features):
        dfs(feature_id)

    return needed


def build_waves(feature_set: Set[str], features: Dict[str, Dict[str, object]]) -> List[List[str]]:
    """Build deterministic parallel waves for a feature subset using Kahn's algorithm."""
    if not feature_set:
        return []

    for fid in sorted(feature_set):
        if fid not in features:
            raise KeyError(f"Unknown feature in feature_set: {fid}")

    indegree: Dict[str, int] = {fid: 0 for fid in feature_set}
    graph: Dict[str, List[str]] = defaultdict(list)

    for fid in sorted(feature_set):
        deps = features[fid].get("deps", [])
        for dep in sorted(deps):
            if dep not in feature_set:
                continue
            graph[dep].append(fid)
            indegree[fid] += 1

    queue = deque(sorted(fid for fid, degree in indegree.items() if degree == 0))
    waves: List[List[str]] = []
    processed = 0

    while queue:
        wave_size = len(queue)
        wave: List[str] = []
        for _ in range(wave_size):
            node = queue.popleft()
            wave.append(node)
            processed += 1
            for nxt in sorted(graph[node]):
                indegree[nxt] -= 1
                if indegree[nxt] == 0:
                    queue.append(nxt)
        waves.append(wave)

    if processed != len(feature_set):
        cyclic = sorted([fid for fid, degree in indegree.items() if degree > 0])
        raise ValueError(f"Cycle detected in feature graph among: {cyclic}")

    return waves
