import typing as t
import time
from .registry import FeatureRegistry

class DAGVisualizer:
    def __init__(self):
        pass
        
    def visualize_plan(self, plan: t.List[t.List[str]], registry: FeatureRegistry):
        print("\n" + "☄️"*30)
        print("🌌 EXECUTION PLAN DAG (Wavefront Analysis)")
        print("☄️"*30)
        
        # calculate max padding
        max_len = 0
        if plan:
            flat_plan = [item for sublist in plan for item in sublist]
            if flat_plan:
                max_len = max(len(f) for f in flat_plan)
        
        for i, level in enumerate(plan):
            print(f"\n🌊 Wave {i+1} [Parallel Context]:")
            for feature in level:
                deps = registry.get_feature(feature).dependencies
                dep_str = f"👈 Depends on: {deps}" if deps else "✨ (Base Feature - Anti-Gravity Origin)"
                print(f"   ├─ 🚀 {feature.ljust(max_len)} {dep_str}")
        print("\n" + "☄️"*30 + "\n")

class ExecutionObserver:
    def __init__(self):
        self.start_time = 0
        
    def start_level(self, level_idx: int, level_features: t.List[str]):
        print(f"[⌛] Executing Wave {level_idx + 1} ({len(level_features)} features in parallel harmony)...")
        
    def feature_completed(self, feature_name: str, success: bool, fallback: bool = False):
        if success and not fallback:
            print(f"  ✅ [SUCCESS]  {feature_name} computed.")
        elif success and fallback:
            print(f"  ⚠️ [FALLBACK] {feature_name} computed using fallback strategy.")
        else:
            print(f"  ❌ [FAILED]   {feature_name} failed catastrophically.")
            
    def execution_finished(self, duration: float, stats: t.Dict[str, t.Any]):
        print("\n" + "="*60)
        print(f"🌠 MISSION ACCOMPLISHED in {duration:.4f}s")
        print("="*60)
        
        success_keys = [f for f, s in stats.items() if s["status"] in ["success", "fallback"]]
        print(f"📈 Total Features Evaluated: {len(stats)}")
        print(f"🎯 Cache Hits / Perfect Reuse: All dependencies automatically shared")
        print(f"⚡ Parallelization Time per Feature:")
        for f in success_keys:
            stat = stats[f]
            is_fb = stat["status"] == "fallback"
            badge = "⚠️" if is_fb else "✅"
            print(f"   {badge} {f}: {stat['dur']:.4f}s")
