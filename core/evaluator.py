from typing import List, Dict, Any

class Evaluator:
    @staticmethod
    def calculate_metrics(
        pipeline_results: List[int], 
        union_size: int, 
        judge_results: List[bool]
    ) -> Dict[str, float]:
        """
        pipeline_results: indices of knowledge points in the union set.
        union_size: total unique points across all pipelines.
        judge_results: boolean list for each point in pipeline output.
        """
        # Recall: Unique points covered by this pipeline / Total union points
        unique_points = set(pipeline_results)
        recall = len(unique_points) / union_size if union_size > 0 else 0.0
        
        # Accuracy: Valid points / Total points output by this pipeline
        valid_count = sum(1 for r in judge_results if r)
        accuracy = valid_count / len(judge_results) if judge_results else 0.0
        
        return {
            "recall": recall,
            "accuracy": accuracy
        }

