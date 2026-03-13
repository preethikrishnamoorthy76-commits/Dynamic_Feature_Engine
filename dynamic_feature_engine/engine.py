import typing as t
import time
from .registry import FeatureRegistry, ModelRegistry
from .planner import ExecutionPlanner, DependencyResolver
from .executor import ParallelExecutor, FeatureCache
from .visualization import ExecutionObserver, DAGVisualizer

class DynamicFeatureEngine:
    """
    The main orchestrator for the Dynamic Feature Execution Engine.
    Combines registry, planner, executor, and magically observable visualizations.
    """
    def __init__(self, max_workers: int = None, use_visualizer: bool = True):
        self.feature_registry = FeatureRegistry()
        self.model_registry = ModelRegistry()
        self.resolver = DependencyResolver(self.feature_registry)
        self.planner = ExecutionPlanner(self.feature_registry, self.resolver)
        self.executor = ParallelExecutor(self.feature_registry, max_workers=max_workers)
        
        self.use_visualizer = use_visualizer
        self.visualizer = DAGVisualizer() if use_visualizer else None
        
    def register_feature(self, name: str, compute_fn: t.Callable, deps: t.List[str] = None, fallback: t.Callable = None, is_streaming: bool = False):
        """Register a feature that magically ties into the DAG"""
        self.feature_registry.register(name, compute_fn, deps, fallback, is_streaming)
        
    def register_model(self, name: str, needs: t.List[str]):
        """Models float above, unaware of the complexity below."""
        self.model_registry.register(name, needs)
        
    def execute(self, request: t.Dict[str, t.Any]) -> t.Dict[str, t.Any]:
        """
        One request, multiple models, everything floats effortlessly.
        request format ex: {"models": ["churn", "fraud"], "customer_id": 12345}
        """
        models = request.get("models", [])
        if not models:
            return {}
            
        # 0. Understand what models require
        target_features = self.model_registry.get_needs(models)
        
        if self.use_visualizer:
            print(f"\n🚀 Engaging Dynamic Execution Engine for Models: {', '.join(models)}")
            print(f"🎯 Target Features: {', '.join(target_features)}")
            
        # 1. Plan Execution Levels (WAVEFRONTS)
        plan = self.planner.create_plan(target_features)
        
        if self.use_visualizer:
            self.visualizer.visualize_plan(plan, self.feature_registry)
            self.executor.set_observer(ExecutionObserver())
            
        # 2. Parallel Execute
        # Pass the request exactly as run_context so computed feature functions can extract IDs, args.
        cache = self.executor.execute_plan(plan, run_context=request)
        
        # 3. Assemble results for each requested model seamlessly Let the magic shine.
        results = {}
        for model in models:
            needs = self.model_registry.get_needs([model])
            # Return dict map for the features target models specifically requested
            results[model] = {feat: cache.get(feat) for feat in needs}
            
        return dict(results)

