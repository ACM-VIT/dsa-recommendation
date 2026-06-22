import os
import json
from datetime import datetime, timezone
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer
from pipeline.recommender.hlr import calculate_urgency
from pipeline.recommender.ranking import rank_candidates
from controllers.submission_controller import user_mastery_store, user_hlr_store
 
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
 
# Load once at startup
client = QdrantClient(path=os.path.join(BASE_DIR, "qdrant_storage_v2"))
model = SentenceTransformer("all-MiniLM-L6-v2")
 
with open(os.path.join(BASE_DIR, "data", "topic_topic_edges_normalized.json"), encoding="utf-8-sig") as f:
    tt_edges = json.load(f)

def handle_get_candidates(topic, min_difficulty, max_difficulty, limit):
    from qdrant_client.models import Filter, FieldCondition, Range
    query_vector = model.encode(topic).tolist()
    results = client.query_points(
        collection_name="problems_v2",
        query=query_vector,
        limit=limit,
        query_filter=Filter(
            must=[
                FieldCondition(
                    key="difficulty_score",
                    range=Range(gte=min_difficulty, lte=max_difficulty)
                )
            ]
        )
    ).points
    return [
        {
            "title_slug": r.payload["title_slug"],
            "title": r.payload["title"],
            "description": r.payload["description"],
            "topics": r.payload["topics"],
            "difficulty_score": r.payload["difficulty_score"],
            "score": r.score
        }
        for r in results
    ]
 
def handle_recommend(user_id, limit):
    mastery = user_mastery_store.get(user_id, {})
    hlr_state = user_hlr_store.get(user_id, {})
    current_time = datetime.now(timezone.utc).timestamp()
 
    topic_scores = {}
    all_topics = set(list(mastery.keys()) + list(hlr_state.keys()))
 
    for topic in all_topics:
        bkt_score = 1 - mastery.get(topic, 0.15)
        urgency = calculate_urgency(hlr_state.get(topic, {}), current_time)
        topic_scores[topic] = 0.6 * bkt_score + 0.4 * urgency
 
    if not topic_scores:
        return {"message": "No history found for user. Start solving problems first.", "candidates": []}
 
    # needs_attention — highest HLR urgency
    urgent_topic = max(
        hlr_state.keys(),
        key=lambda t: calculate_urgency(hlr_state[t], current_time)
    ) if hlr_state else None
 
    # weak_topic should be different from urgent_topic
    weak_topic = min(
    [t for t in mastery.keys() if t != urgent_topic],
    key=lambda t: mastery[t]
     ) if mastery and any(t != urgent_topic for t in mastery.keys()) else None
    # current_topic — highest combined score excluding above two
    remaining = [t for t in topic_scores if t != urgent_topic and t != weak_topic]
    current_topic = max(remaining, key=topic_scores.get) if remaining else None
 
    topic_categories = {}
    if urgent_topic:
        topic_categories[urgent_topic] = "needs_attention"
    if weak_topic:
        topic_categories[weak_topic] = "weak_topic"
    if current_topic:
        topic_categories[current_topic] = "current_topic"
 
    top_topics = [t for t in [urgent_topic, weak_topic, current_topic] if t]
 
    seen = set()
    candidates = []
    per_topic = max(1, limit // max(1, len(top_topics)))
 
    for topic in top_topics:
        query_vector = model.encode(topic).tolist()
        results = client.query_points(
            collection_name="problems_v2",
            query=query_vector,
            limit=per_topic
        ).points
 
        for r in results:
            slug = r.payload["title_slug"]
            if slug not in seen:
                seen.add(slug)
                candidates.append({
                    "title_slug": slug,
                    "title": r.payload["title"],
                    "description": r.payload["description"],
                    "topics": r.payload["topics"],
                    "difficulty_score": r.payload["difficulty_score"],
                    "score": r.score,
                    "category": topic_categories.get(topic, "general")
                })
 
    ranked = rank_candidates(
        candidates=candidates,
        user_bkt_mastery=mastery,
        user_hlr_state=hlr_state,
        recent_topics=list(mastery.keys())[-10:],
        topic_topic_edges=tt_edges,
        current_timestamp=current_time
    )
 
    return {
        "userId": user_id,
        "topic_breakdown": {
            "needs_attention": urgent_topic,
            "weak_topic": weak_topic,
            "current_topic": current_topic
        },
        "recommended": ranked[0] if ranked else None,
        "all_candidates_ranked": ranked
    }
 