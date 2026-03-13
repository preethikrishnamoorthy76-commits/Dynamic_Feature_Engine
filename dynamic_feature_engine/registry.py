import typing as t
from dataclasses import dataclass, field

@dataclass
class FeatureNode:
    name: str
    compute_fn: t.Callable
    dependencies: t.List[str] = field(default_factory=list)
    fallback_fn: t.Optional[t.Callable] = None
    is_streaming: bool = False

class FeatureRegistry:
    def __init__(self):
        self.features: t.Dict[str, FeatureNode] = {}
        
    def register(self, name: str, compute_fn: t.Callable, deps: t.List[str] = None, fallback_fn: t.Optional[t.Callable] = None, is_streaming: bool = False):
        if deps is None:
            deps = []
        self.features[name] = FeatureNode(name, compute_fn, deps, fallback_fn, is_streaming)
        self._validate_acyclic()
        
    def _validate_acyclic(self):
        # Build graph and check for cycles using DFS
        visited = set()
        path = set()
        
        def dfs(node):
            if node in path:
                raise ValueError(f"Cycle detected involving feature: {node}")
            if node in visited:
                return
            visited.add(node)
            path.add(node)
            
            # Check dependencies
            if node in self.features:
                for dep in self.features[node].dependencies:
                    dfs(dep)
            path.remove(node)
            
        for feature in self.features:
            dfs(feature)
            
    def get_feature(self, name: str) -> FeatureNode:
        if name not in self.features:
            raise KeyError(f"Feature '{name}' not found in registry. Ensure it is registered and no typos exist.")
        return self.features[name]

class ModelRegistry:
    def __init__(self):
        self.models: t.Dict[str, t.List[str]] = {}
        
    def register(self, model_name: str, needs: t.List[str]):
        """Models float above, unaware of the complexity below."""
        self.models[model_name] = needs
        
    def get_needs(self, model_names: t.List[str]) -> t.Set[str]:
        needs = set()
        for m in model_names:
            if m not in self.models:
                raise KeyError(f"Model '{m}' not found in registry.")
            needs.update(self.models[m])
        return needs
