from openai import OpenAI
import numpy as np
import json
import random

with open("api.json") as f:
    api_data = json.load(f)

client = OpenAI(
    base_url="https://integrate.api.nvidia.com/v1",
    api_key=random.choice(api_data["api_keys"])
)

def embed(texts):
    response = client.embeddings.create(
        input=texts,
        model="nvidia/llama-3.2-nv-embedqa-1b-v2",
        encoding_format="float",
        extra_body={"input_type": "query", "truncate": "NONE"}
    )
    return [item.embedding for item in response.data]

def cosine_sim(a, b):
    a = np.array(a)
    b = np.array(b)
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def dedupe_questions(questions, threshold=0.93):
    embeddings = embed(questions)

    kept = []
    kept_embs = []

    for q, e in zip(questions, embeddings):
        is_duplicate = False
        for ke, ke_emb in zip(kept, kept_embs):
            if cosine_sim(e, ke_emb) >= threshold:
                print(f"去重: {q} vs {ke}, 相似度 {cosine_sim(e, ke_emb):.4f} >= {threshold}")
                is_duplicate = True
                break
        if not is_duplicate:
            kept.append(q)
            kept_embs.append(e)

    return kept

# ---- 你直接调用就行 ----
questions = [
    "What is the capital of France?",
    "Which city is the capital of France?",
    "Where is the capital city of France located?",
    "Explain the theory of relativity.",
    "What does Einstein's relativity theory mean?",
    "How to make fried rice?"
]

deduped = dedupe_questions(questions, threshold=0.93)
print("\n去重后：")
for q in deduped:
    print("-", q)
