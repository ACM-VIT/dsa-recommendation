import json
from collections import defaultdict
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, VectorParams, Distance, Filter, FieldCondition, Range

import os
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
REPO_ROOT = BASE_DIR
with open(os.path.join(BASE_DIR, "data", "problem_nodes_normalized.json")) as f:
    problem_nodes = json.load(f)
with open(os.path.join(BASE_DIR, "data", "problem_topic_edges_normalized.json")) as f:
    pt_edges = json.load(f)
with open(os.path.join(BASE_DIR, "data", "topic_nodes_normalized.json")) as f:
    topic_nodes = json.load(f)
with open(os.path.join(BASE_DIR, "data", "topic_topic_edges_normalized.json"), encoding="utf-8-sig") as f:
    tt_edges = json.load(f)

print(f"Problems: {len(problem_nodes)}")
print(f"Topics: {len(topic_nodes)}")
print(f"Problem-Topic edges: {len(pt_edges)}")
print(f"Topic-Topic edges: {len(tt_edges)}")

problem_topics = defaultdict(list)
for edge in pt_edges:
    problem_topics[edge["source"]].append(edge["target"])

problems = []
for p in problem_nodes:
    slug = p["title_slug"]
    topics = problem_topics.get(slug, [])
    # When building each problem dict
    problem = {
    "title_slug": slug,
    "title": p["title"],
    "description": p["description"],
    "topics": topics,
    "difficulty_score": p.get("difficulty_score", 0),  # now reads from JSON
    "sample_test_cases": p.get("sample_test_cases", [])
}
    problems.append(problem)

print(f"Loaded {len(problems)} problems with topics")

def create_embed_text(problem):
    topics_str = " ".join(problem["topics"])
    text = f"{problem['description'][:500]} Topics: {topics_str}"
    return text

model = SentenceTransformer("all-MiniLM-L6-v2")
texts = [create_embed_text(p) for p in problems]
embeddings = model.encode(texts, batch_size=64, show_progress_bar=True)

for i, problem in enumerate(problems):
    problem["embedding"] = embeddings[i].tolist()

client = QdrantClient(path=os.path.join(REPO_ROOT, "qdrant_storage_v2"))

existing = [c.name for c in client.get_collections().collections]
if "problems_v2" in existing:
    client.delete_collection("problems_v2")

client.create_collection(
    collection_name="problems_v2",
    vectors_config=VectorParams(size=384, distance=Distance.COSINE)
)

points = []
for i, problem in enumerate(problems):
    points.append(
        PointStruct(
            id=i,
            vector=problem["embedding"],
            payload={
    "title_slug": problem["title_slug"],
    "title": problem["title"],
    "description": problem["description"][:300],
    "topics": problem["topics"],
    "difficulty_score": problem["difficulty_score"],  # will be 0 until preprocessing adds it
}
        )
    )

batch_size = 100
for i in range(0, len(points), batch_size):
    client.upsert(
        collection_name="problems_v2",
        points=points[i:i+batch_size]
    )
    print(f"Uploaded {min(i+batch_size, len(points))} / {len(points)}")

test_queries = [
    "find two numbers that sum to target in array",
    "binary search on sorted array",
    "graph traversal BFS DFS",
    "dynamic programming longest subsequence",
    "sliding window maximum subarray"
]

for q in test_queries:
    print(f"\nQuery: {q}")
    query_vector = model.encode(q).tolist()
    results = client.query_points(
        collection_name="problems_v2",
        query=query_vector,
        limit=3
    ).points
    for r in results:
        print(f"  {r.payload['title']} | topics: {r.payload['topics'][:3]} | score: {r.score:.3f}")

client.close()

