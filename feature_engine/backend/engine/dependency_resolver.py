from __future__ import annotations

from collections import defaultdict, deque
from typing import Dict, List, Set

from backend.features.registry import FeatureRegistry


class DependencyResolver:
    def __init__(self, registry: FeatureRegistry) -> None:
        self.registry = registry

    def _collect_needed_features(self, required_features: List[str]) -> Set[str]:
        needed: Set[str] = set()

        def dfs(feature_name: str) -> None:
            if feature_name in needed:
                return
            definition = self.registry.get(feature_name)
            needed.add(feature_name)
            for dependency in definition.depends_on:
                dfs(dependency)

        for feature in required_features:
            dfs(feature)

        return needed

    def resolve_waves(self, required_features: List[str]) -> List[List[str]]:
        if not required_features:
            return []

        needed = self._collect_needed_features(required_features)
        indegree: Dict[str, int] = {feature: 0 for feature in needed}
        graph: Dict[str, List[str]] = defaultdict(list)

        for feature in needed:
            definition = self.registry.get(feature)
            for dependency in definition.depends_on:
                if dependency not in needed:
                    continue
                graph[dependency].append(feature)
                indegree[feature] += 1

        queue = deque(sorted([name for name, degree in indegree.items() if degree == 0]))
        waves: List[List[str]] = []
        processed = 0

        while queue:
            current_wave_size = len(queue)
            wave: List[str] = []
            for _ in range(current_wave_size):
                node = queue.popleft()
                wave.append(node)
                processed += 1
                for neighbor in sorted(graph[node]):
                    indegree[neighbor] -= 1
                    if indegree[neighbor] == 0:
                        queue.append(neighbor)
            waves.append(wave)

        if processed != len(needed):
            unresolved = [name for name, degree in indegree.items() if degree > 0]
            raise ValueError(f"Circular dependency detected among features: {unresolved}")

        return waves
