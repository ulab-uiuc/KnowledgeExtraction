import json
import pickle
import numpy as np

MODELS = ["llama-3.1-8b-instruct", "llama-3.1-70b-instruct", "llama-3.1-405b-instruct"]
DOMAIN = "Transformer_Architectures"

print("=== Diagnosing Merge Logic ===\n")

# Simulate the exact merge logic
union_nodes = []
STRICT_MATCH = 0.92

for model in MODELS:
    raw_path = f"results/model_comparison_l5w5/{DOMAIN}/{model}/P4_Taxonomy_L5W5_raw.json"
    emb_path = f"results/model_comparison_l5w5/{DOMAIN}/{model}/P4_Taxonomy_L5W5.emb.pkl"
    
    with open(raw_path, "r") as f:
        points = json.load(f)
    with open(emb_path, "rb") as f:
        emb_dict = pickle.load(f)
    embeddings = np.array([emb_dict[p] for p in points])
    
    N = len(points)
    print(f"\n{model}: {N} raw points")
    
    # Simulate Line 80
    norms = np.stack([e / (np.linalg.norm(e) + 1e-9) for e in embeddings])
    
    # Simulate Line 83
    assignments = [None] * N
    
    # Simulate Line 86-94 (strict matching against union)
    if union_nodes:
        union_mat = np.stack([n["emb"] / (np.linalg.norm(n["emb"]) + 1e-9) for n in union_nodes])
        sim_to_union = np.dot(norms, union_mat.T)
        
        strict_merged = 0
        for i in range(N):
            max_idx = np.argmax(sim_to_union[i])
            if sim_to_union[i, max_idx] >= STRICT_MATCH:
                assignments[i] = max_idx
                strict_merged += 1
        print(f"  Strict merged (>=0.92): {strict_merged}")
    
    # Count how many are still None (will be new nodes potentially)
    new_indices = [i for i in range(N) if assignments[i] is None]
    print(f"  Potentially new points: {len(new_indices)}")
    
    # SIMULATE INTERNAL DEDUP (Line 114-147)
    internal_assignments = {}  # sub_idx -> representative_sub_idx
    if len(new_indices) > 1:
        sub_norms = norms[new_indices]
        internal_sims = np.dot(sub_norms, sub_norms.T)
        
        for i in range(len(new_indices)):
            if i in internal_assignments: continue
            prev_sims = internal_sims[i, :i]
            if len(prev_sims) == 0: continue
            
            m_idx = np.argmax(prev_sims)
            if prev_sims[m_idx] >= STRICT_MATCH:
                internal_assignments[i] = m_idx
        
        print(f"  Internal duplicates found: {len(internal_assignments)}")
    
    # FIXED: Create new nodes first (only for non-duplicates)
    for i in range(len(new_indices)):
        real_idx = new_indices[i]
        if i not in internal_assignments:
            u_idx = len(union_nodes)
            union_nodes.append({"text": points[real_idx], "emb": embeddings[real_idx], "models": set()})
            assignments[real_idx] = u_idx
    
    # Handle internal duplicates (after representatives are created)
    for i in range(len(new_indices)):
        real_idx = new_indices[i]
        if i in internal_assignments:
            rep_sub_idx = internal_assignments[i]
            rep_real_idx = new_indices[rep_sub_idx]
            assignments[real_idx] = assignments[rep_real_idx]
    
    # FIXED: Update ownership using enumerate
    print(f"  Before ownership update: union has {len(union_nodes)} nodes")
    
    # Count how many assignments are valid
    valid_assignments = sum(1 for a in assignments if a is not None)
    print(f"  Valid assignments: {valid_assignments}/{N}")
    
    # Actually update (FIXED version)
    for i, a_idx in enumerate(assignments):
        if a_idx is not None:
            union_nodes[a_idx]["models"].add(model)
    
    # Check ownership
    owned_count = sum(1 for n in union_nodes if model in n["models"])
    print(f"  After ownership update: {owned_count} nodes have {model} in models")
    print(f"  Expected: {N} (all points should be owned)")
    print(f"  ❌ LOST: {N - owned_count} points not owned!" if owned_count < N else "  ✓ OK")

print(f"\n{'='*60}")
print(f"Final union size: {len(union_nodes)}")
for model in MODELS:
    count = sum(1 for n in union_nodes if model in n["models"])
    print(f"{model}: owns {count} nodes")
