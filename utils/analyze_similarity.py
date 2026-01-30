import asyncio
import json
import os
import numpy as np
from core.processor import KnowledgeProcessor
from agents.clientpool import MultiKeyClientPool

async def analyze_similarity(results_dir):
    # 1. Load configuration
    if not os.path.exists("api.json"):
        print("Error: api.json not found")
        return
        
    with open("api.json") as f:
        api_data = json.load(f)
    
    api_keys = api_data["api_keys"]
    
    # Check for embedding config
    is_code = "code" in results_dir.lower()
    config_key = "code_embed_config" if is_code else "embed_config"
    embed_config = api_data.get(config_key, api_data.get("embed_config", {}))
    
    if not embed_config:
        print(f"Error: No {config_key} configuration found in api.json")
        return

    # 2. Initialize processor
    client_pool = MultiKeyClientPool(api_keys=api_keys)
    embed_client_pool = MultiKeyClientPool(
        api_keys=embed_config.get("api_keys", api_keys),
        base_url=embed_config.get("base_url")
    )
    
    processor = KnowledgeProcessor(
        client_pool=client_pool,
        embed_client_pool=embed_client_pool,
        embed_model=embed_config.get("model")
    )

    # 3. Read all raw points
    if not os.path.exists(results_dir):
        print(f"Error: Directory {results_dir} does not exist. Run main.py first.")
        return

    unique_points = set()
    for f in os.listdir(results_dir):
        if f.endswith("_raw.json"):
            with open(os.path.join(results_dir, f), "r") as f_in:
                points = json.load(f_in)
                unique_points.update(points)
    
    unique_points = list(unique_points)
    print(f"Total unique knowledge points found: {len(unique_points)}")
    
    if len(unique_points) < 2:
        print("Too few points to perform comparison.")
        return

    # 4. Get Embeddings
    print(f"Calling {embed_config.get('model')} to fetch vectors...")
    embeddings = await processor.get_embeddings(unique_points)
    
    # 5. Calculate pairwise similarity
    print("Calculating pairwise cosine similarity distribution...")
    embs_matrix = np.stack(embeddings)
    # Normalize
    norms = np.linalg.norm(embs_matrix, axis=1, keepdims=True)
    embs_matrix = embs_matrix / (norms + 1e-9)
    
    # Compute similarity matrix
    sim_matrix = np.dot(embs_matrix, embs_matrix.T)
    
    # Get upper triangle only (excluding self-similarity)
    indices = np.triu_indices(len(unique_points), k=1)
    similarities = sim_matrix[indices]

    # 6. Output analysis report
    print("\n" + "="*50)
    print(f"SIMILARITY ANALYSIS REPORT: {results_dir}")
    print("="*50)
    print(f"Pairs compared:     {len(similarities)}")
    print(f"Min similarity:     {np.min(similarities):.4f}")
    print(f"Max similarity:     {np.max(similarities):.4f}")
    print(f"Mean similarity:    {np.mean(similarities):.4f}")
    print(f"Median similarity:  {np.median(similarities):.4f}")
    
    print("\nPercentile Distribution:")
    for p in [50, 75, 90, 95, 99]:
        print(f"  {p}th Percentile: {np.percentile(similarities, p):.4f}")
        
    print("\nRange Distribution:")
    ranges = [(0, 0.5), (0.5, 0.7), (0.7, 0.8), (0.8, 0.85), (0.85, 0.9), (0.9, 0.95), (0.95, 1.0)]
    for low, high in ranges:
        count = np.sum((similarities >= low) & (similarities < high))
        print(f"  [{low:.2f} - {high:.2f}): {count:5d} pairs ({count/len(similarities):.2%})")
    
    print("\nRecommendation: Observe the percentage above 0.85. Highly discriminative models show fewer high-score pairs.")
    print("="*50)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", type=str, required=True, help="Results directory to analyze")
    args = parser.parse_args()
    
    asyncio.run(analyze_similarity(args.dir))
