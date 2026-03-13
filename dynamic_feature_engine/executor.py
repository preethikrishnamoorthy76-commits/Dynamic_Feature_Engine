import typing as t
import threading
import time
import inspect
from concurrent.futures import ThreadPoolExecutor, as_completed
from .registry import FeatureRegistry

class FeatureCache:
    def __init__(self):
        self.cache: t.Dict[str, t.Any] = {}
        self.lock = threading.Lock()
        
    def get(self, key: str) -> t.Any:
        with self.lock:
            return self.cache.get(key)
            
    def set(self, key: str, value: t.Any):
        with self.lock:
            self.cache[key] = value
            
    def contains(self, key: str) -> bool:
        with self.lock:
            return key in self.cache

class ParallelExecutor:
    def __init__(self, registry: FeatureRegistry, max_workers: int = None):
        self.registry = registry
        self.max_workers = max_workers
        self.observer = None
        
    def set_observer(self, observer):
        self.observer = observer
        
    def execute_plan(self, plan: t.List[t.List[str]], run_context: t.Dict[str, t.Any]) -> FeatureCache:
        """
        Executes the plan level by level. All features in a level are completely independent
        and thus run in perfect parallel harmony, waiting for nothing but the levels before them.
        """
        cache = FeatureCache()
        stats = {}
        start_time = time.time()
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            for level_idx, level in enumerate(plan):
                if self.observer:
                    self.observer.start_level(level_idx, level)
                    
                futures_to_feature = {}
                for feature_name in level:
                    if cache.contains(feature_name):
                        continue
                        
                    future = executor.submit(self._compute_feature, feature_name, cache, run_context)
                    futures_to_feature[future] = feature_name
                    
                for future in as_completed(futures_to_feature):
                    feature_name = futures_to_feature[future]
                    t_start = time.time()
                    try:
                        result = future.result()
                        cache.set(feature_name, result)
                        stats[feature_name] = {"status": "success", "dur": time.time() - t_start}
                        if self.observer:
                            self.observer.feature_completed(feature_name, success=True)
                    except Exception as e:
                        # Attempt fallback strategy gracefully
                        feature_node = self.registry.get_feature(feature_name)
                        if feature_node.fallback_fn:
                            try:
                                result = self._invoke_fn(feature_node.fallback_fn, feature_node, cache, run_context)
                                cache.set(feature_name, result)
                                stats[feature_name] = {"status": "fallback", "dur": time.time() - t_start}
                                if self.observer:
                                    self.observer.feature_completed(feature_name, success=True, fallback=True)
                            except Exception as fallback_e:
                                stats[feature_name] = {"status": "failed", "error": str(fallback_e)}
                                if self.observer:
                                    self.observer.feature_completed(feature_name, success=False)
                                raise RuntimeError(f"Feature '{feature_name}' and its fallback failed: {e} -> {fallback_e}")
                        else:
                            stats[feature_name] = {"status": "failed", "error": str(e)}
                            if self.observer:
                                self.observer.feature_completed(feature_name, success=False)
                            raise RuntimeError(f"Feature '{feature_name}' failed with no fallback strategy: {e}")
                            
        if self.observer:
            self.observer.execution_finished(time.time() - start_time, stats)
            
        return cache

    def _invoke_fn(self, fn: t.Callable, feature_node, cache: FeatureCache, run_context: t.Dict[str, t.Any]):
        """
        Helper to invoke a function by resolving its kwargs properly based on its signature.
        It intelligently maps cached dependencies and the run context to the function's parameters.
        """
        sig = inspect.signature(fn)
        kwargs = {}
        
        for param_name in sig.parameters:
            if param_name == 'context':
                kwargs['context'] = run_context
            elif cache.contains(param_name):
                kwargs[param_name] = cache.get(param_name)
            elif param_name in feature_node.dependencies:
                # If it's a declared dependency but not in cache for some reason (shouldn't happen in normal flow)
                kwargs[param_name] = cache.get(param_name)
            # Optional: handle parameters that might be in run_context but not explicitly 'context'
            elif param_name in run_context:
                 kwargs[param_name] = run_context[param_name]
        
        return fn(**kwargs)

    def _compute_feature(self, feature_name: str, cache: FeatureCache, run_context: t.Dict[str, t.Any]):
        feature_node = self.registry.get_feature(feature_name)
        return self._invoke_fn(feature_node.compute_fn, feature_node, cache, run_context)
