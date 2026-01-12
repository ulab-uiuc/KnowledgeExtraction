from __future__ import annotations
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Tuple
import numpy as np
from core.cleaner import KnowledgeCleaner

class BasePipeline(ABC):
    def __init__(self, agent, processor, model: str = "meta/llama-3.2-3b-instruct"):
        self.agent = agent
        self.processor = processor # We need processor for embeddings
        self.model = model
        self.saturation_threshold = 0.92 # Similarity above this is "old"
        self.min_growth_ratio = 0.01     # If Growth < 1%, consider potentially saturated
        self.min_efficiency_ratio = 0.1   # If Efficiency < 10%, consider potentially saturated

    @abstractmethod
    async def get_next_step(self, query: str, history: List[str], current_points: List[str], turn: int) -> str:
        """
        Subclasses implement the specific prompting strategy for the next turn.
        """
        pass

    async def run(self, query: str) -> List[str]:
        """
        The common saturation-based execution logic.
        """
        all_raw_points = []
        all_embeddings = []
        history = []
        turn = 1
        patience = 1
        stale_turns = 0
        
        print(f"  Starting saturation run for {self.__class__.__name__}...")
        
        while True:
            # 1. Generate content for this turn (pass turn number)
            # Create a user prompt for this turn to keep history consistent
            response = await self.get_next_step(query, history, all_raw_points, turn)
            
            # Subclasses handle their own internal prompting, but we need to record 
            # the fact that a turn happened in history. 
            # To do this correctly, get_next_step should return the response,
            # and we should capture what prompt was used.
            # However, for simplicity and to avoid changing the interface too much,
            # we will let the subclasses manage history appending if they need custom User prompts,
            # OR we just append the Assistant response here.
            
            new_points = KnowledgeCleaner.clean_bullet_points(response)
            
            if not new_points:
                print(f"    Turn {turn}: No points extracted. Stopping.")
                break
                
            # 2. Check for novelty
            new_embeddings = await self.processor.get_embeddings(new_points)
            
            novel_points_this_turn = []
            novel_embeddings_this_turn = []
            
            for p, emb in zip(new_points, new_embeddings):
                is_novel = True
                # Combine historical embeddings with already identified novel embeddings from THIS turn
                # to prevent intra-turn duplicates from being counted as novel.
                comparison_pool = all_embeddings + novel_embeddings_this_turn
                
                if comparison_pool:
                    # Find max similarity with any point from previous turns OR current turn's novel points
                    similarities = [self.processor.cosine_similarity(emb, prev_emb) for prev_emb in comparison_pool]
                    if max(similarities) > self.saturation_threshold:
                        is_novel = False
                
                if is_novel:
                    novel_points_this_turn.append(p)
                    novel_embeddings_this_turn.append(emb)
            
            novel_count = len(novel_points_this_turn)
            # Calculate growth rate relative to all points found so far
            novel_ratio = novel_count / len(all_raw_points) if all_raw_points else 1.0
            # Calculate efficiency of this turn
            efficiency = novel_count / len(new_points)
            
            print(f"    Turn {turn}: Found {len(new_points)} points. Novel: {novel_count} (Growth: {novel_ratio:.2%}, Eff: {efficiency:.1%})")
            
            # Store everything
            all_raw_points.extend(new_points)
            all_embeddings.extend(new_embeddings)
            
            # Update history with the assistant response
            # Note: The 'User' part is handled inside get_next_step's internal logic for now
            history.append(f"Assistant: {response}")
            
            # 3. Saturation Check
            # Growth < 1% OR Efficiency < 10% OR absolute novel count is very low
            if novel_ratio < self.min_growth_ratio or efficiency < self.min_efficiency_ratio or novel_count < 5:
                stale_turns += 1
                if stale_turns > patience:
                    print(f"    Saturated after {turn} turns.")
                    break
            else:
                stale_turns = 0 # Reset if we find new stuff
            
            turn += 1
            if turn > 15: # Safety cap
                print("    Reached safety limit of 20 turns.")
                break
                
        # Deduplicate raw points by string exact match before returning
        return list(dict.fromkeys(all_raw_points))
