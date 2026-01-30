from __future__ import annotations
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Tuple, Callable, Optional
import numpy as np
import time
from core.cleaner import KnowledgeCleaner

class BasePipeline(ABC):
    def __init__(self, agent, processor, model: str = "meta/llama-3.1-70b-instruct"):
        self.agent = agent
        self.processor = processor 
        self.model = model
        self.saturation_threshold = 0.92 
        self.min_growth_ratio = 0.01     
        self.min_efficiency_ratio = 0.1   
        self.save_callback: Callable = None
        
        # Token and Trajectory state
        self.start_gen_tokens = 0
        self.start_embed_tokens = 0
        self.trajectory = []

    def set_callback(self, callback: Callable):
        self.save_callback = callback

    def get_internal_state(self) -> Dict[str, Any]:
        return {}

    def set_internal_state(self, state: Dict[str, Any]):
        pass

    async def record_snapshot(self, turn_label: Any, current_points: List[str], history: List[str], novel_count: int = 0, new_count: int = 0):
        cumulative_gen = self.agent.total_tokens - self.start_gen_tokens
        cumulative_embed = self.processor.total_tokens - self.start_embed_tokens
        
        meta = {
            "turn": turn_label,
            "cumulative_tokens": max(0, cumulative_gen + cumulative_embed),
            "cumulative_gen_tokens": max(0, cumulative_gen),
            "cumulative_embed_tokens": max(0, cumulative_embed),
            "timestamp": time.time(),
            "new_points_count": new_count,
            "novel_points_count": novel_count
        }
        
        snapshot = {
            "meta": meta,
            "points": list(current_points),  # CRITICAL: Create a copy to prevent all snapshots sharing the same list reference
            "history": list(history),  # Same for history
            "internal_state": self.get_internal_state()
        }
        
        self.trajectory.append(snapshot)
        if self.save_callback:
            await self.save_callback(self.trajectory)

    @abstractmethod
    async def get_next_step(self, query: str, history: List[str], current_points: List[str], turn: int) -> str:
        pass

    async def run(self, query: str, resume_trajectory: List[Dict] = None, max_turns: int = 30, silent: bool = True) -> Dict[str, Any]:
        all_raw_points = []
        all_embeddings = []
        history = []
        turn = 1
        stale_turns = 0
        patience = 1
        
        self.start_gen_tokens = self.agent.total_tokens
        self.start_embed_tokens = self.processor.total_tokens
        self.trajectory = []

        if resume_trajectory:
            last_snapshot = resume_trajectory[-1]
            all_raw_points = list(last_snapshot["points"])
            history = list(last_snapshot["history"])
            self.trajectory = resume_trajectory
            self.set_internal_state(last_snapshot.get("internal_state", {}))
            
            self.start_gen_tokens -= last_snapshot["meta"].get("cumulative_gen_tokens", 0)
            self.start_embed_tokens -= last_snapshot["meta"].get("cumulative_embed_tokens", 0)
            
            if all_raw_points and not silent:
                print(f"  [Resume] Loading and cleaning {len(all_raw_points)} points...")
                # ... rest of resume log ...
            
            # ... calculate turn ...
            last_turn = last_snapshot["meta"]["turn"]
            if last_turn == "init": turn = 1
            elif isinstance(last_turn, int): turn = last_turn + 1
            else: turn = 1
            if not silent: print(f"  Resuming from Turn {turn}...")
        else:
            if not silent: print(f"  Starting saturation run for {self.__class__.__name__}...")
            await self.record_snapshot(0, [], history)

        while True:
            if not silent: print(f"    Turn {turn}: LLM Requesting...", end="", flush=True)
            response = await self.get_next_step(query, history, all_raw_points, turn)
            if not response: break
            
            new_raw = KnowledgeCleaner.clean_bullet_points(response, min_length=10)
            if not silent: print(f" Found {len(new_raw)} raw items.", end="", flush=True)
            
            if not new_raw:
                # Empty output also counts as stale turn, subject to patience
                if not silent: print(f" Novel: 0 (empty after cleaning)", flush=True)
                history.append(f"Assistant: {response}")  # CRITICAL: Update history even when empty
                await self.record_snapshot(turn, all_raw_points, history, 0, 0)
                stale_turns += 1
                if stale_turns > patience:
                    if not silent: print(f"    Saturated after {turn} turns (empty output).", flush=True)
                    break
                turn += 1
                if turn > max_turns: break
                continue
                
            new_embeddings = await self.processor.get_embeddings(new_raw)
            is_novel_mask = self.processor.get_novelty_mask(new_embeddings, all_embeddings, self.saturation_threshold)
            
            novel_points = [p for i, p in enumerate(new_raw) if is_novel_mask[i]]
            novel_embeddings = [e for i, e in enumerate(new_embeddings) if is_novel_mask[i]]
            novel_count = len(novel_points)
            
            growth = novel_count / len(all_raw_points) if all_raw_points else 1.0
            efficiency = novel_count / len(new_raw)
            if not silent: print(f" Novel: {novel_count} (Growth: {growth:.2%}, Eff: {efficiency:.1%})", flush=True)
            
            if novel_count > 0:
                all_raw_points.extend(novel_points)
                all_embeddings.extend(novel_embeddings)
            
            history.append(f"Assistant: {response}")
            await self.record_snapshot(turn, all_raw_points, history, novel_count, len(new_raw))

            if growth < self.min_growth_ratio or efficiency < self.min_efficiency_ratio or novel_count < 3:
                stale_turns += 1
                if stale_turns > patience:
                    if not silent: print(f"    Saturated after {turn} turns.", flush=True)
                    break
            else:
                stale_turns = 0 
            
            turn += 1
            if turn > max_turns: break
        
        if self.trajectory:
            self.trajectory[-1]["meta"]["is_completed"] = True
            if self.save_callback: await self.save_callback(self.trajectory)

        return {
            "points": all_raw_points,
            "trajectory": self.trajectory
        }
