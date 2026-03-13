import typing as t
from collections import deque
from .registry import FeatureRegistry

class DependencyResolver:
    def __init__(self, registry: FeatureRegistry):
        self.registry = registry
        
    def resolve_requirements(self, target_features: t.Set[str]) -> t.Set[str]:
        """Finds all required features to compute target features."""
        required = set()
        queue = deque(target_features)
        
        while queue:
            node = queue.popleft()
            if node not in required:
                required.add(node)
                feature = self.registry.get_feature(node)
                for dep in feature.dependencies:
                    if dep not in required:
                        queue.append(dep)
        return required
        
class ExecutionPlanner:
    def __init__(self, registry: FeatureRegistry, resolver: DependencyResolver):
        self.registry = registry
        self.resolver = resolver
        
    def create_plan(self, target_features: t.Set[str]) -> t.List[t.List[str]]:
        """
        Creates a topological execution plan (list of parallel levels).
        Lowest level features (no dependencies) first.
        This allows maximum safe parallelism.
        """
        required_features = self.resolver.resolve_requirements(target_features)
        
        # Build indegree map and graph
        in_degree = {f: 0 for f in required_features}
        graph = {f: [] for f in required_features}
        
        for feature_name in required_features:
            feature_node = self.registry.get_feature(feature_name)
            for dep in feature_node.dependencies:
                if dep in required_features:
                    # Dep must be computed BEFORE feature_name
                    graph[dep].append(feature_name)
                    in_degree[feature_name] += 1
                    
        # Topological Sort into levels (Kahn's algorithm modified for levels)
        queue = deque([f for f, deg in in_degree.items() if deg == 0])
        levels = []
        
        while queue:
            level_size = len(queue)
            current_level = []
            for _ in range(level_size):
                curr = queue.popleft()
                current_level.append(curr)
                
                for neighbor in graph[curr]:
                    in_degree[neighbor] -= 1
                    if in_degree[neighbor] == 0:
                        queue.append(neighbor)
            levels.append(current_level)
            
        if sum(len(lvl) for lvl in levels) != len(required_features):
            raise ValueError("Cycle detected during plan creation! Check feature dependencies.")
            
        return levels
