import json
from datetime import datetime, timezone
from pipeline.recommender.hlr import calculate_urgency

WEIGHTS = {
    "bkt_mastery":    0.35,  # how well user knows this topic
    "hlr_urgency":    0.25,  # how urgently topic needs review
    "similarity":     0.25,  # how relevant problem is to topic query
    "variety":        0.15,  # penalise recently seen topics
}
def calculate_variety_score(problem_topics, recent_topics, window=10):
   
    if not recent_topics:
        return 1.0

    recent_count = sum(
        1 for t in problem_topics if t in recent_topics[-window:]
    )
    variety_score = 1.0 - (recent_count / max(1, len(problem_topics)))
    return round(max(0.0, variety_score), 4)

def prerequisite_check(problem_topics, user_bkt_mastery, topic_topic_edges, threshold=0.75):
    # TopicPrerequisite table exists in PostgreSQL with proper prerequisite edges
    # Hard block disabled until backend provides DB connection
    # TODO: query TopicPrerequisite table once PostgreSQL connection is available
    return True

def rank_candidates(
    candidates,
    user_bkt_mastery,
    user_hlr_state,
    recent_topics,
    topic_topic_edges,
    current_timestamp=None
):
    
    if current_timestamp is None:
        current_timestamp = datetime.now(timezone.utc).timestamp()

    ranked = []

    for candidate in candidates:
        problem_topics = candidate.get("topics", [])
        if not problem_topics:
             continue
        similarity_score = candidate.get("score", 0.0)

        # Hard block — skip if prerequisites not met
        if not prerequisite_check(problem_topics, user_bkt_mastery, topic_topic_edges):
            continue

        # 1. BKT mastery score
        # Average mastery across all topics of this problem
        # Low mastery = higher priority (needs practice)
        topic_masteries = [user_bkt_mastery.get(t, 0.15) for t in problem_topics]
        avg_mastery = sum(topic_masteries) / max(1, len(topic_masteries))
        # Invert — low mastery = high score = higher priority
        bkt_score = 1.0 - avg_mastery

        # 2. HLR urgency score
        # Average urgency across all topics of this problem
        topic_urgencies = [
            calculate_urgency(user_hlr_state.get(t, {}), current_timestamp)
            for t in problem_topics
        ]
        hlr_score = sum(topic_urgencies) / max(1, len(topic_urgencies))

        # 3. Similarity score from vector pool
        # Already between 0 and 1
        sim_score = similarity_score

        # 4. Variety score
        variety_score = calculate_variety_score(problem_topics, recent_topics)

        # Final weighted score
        final_score = (
            WEIGHTS["bkt_mastery"] * bkt_score +
            WEIGHTS["hlr_urgency"] * hlr_score +
            WEIGHTS["similarity"]  * sim_score +
            WEIGHTS["variety"]     * variety_score
        )

        ranked.append({
           "title_slug": candidate.get("title_slug"),
           "title": candidate.get("title"),
           "description": candidate.get("description"),
           "topics": problem_topics,
           "difficulty_score": candidate.get("difficulty_score"),
           "category": candidate.get("category", "general"),
           "bkt_score": round(bkt_score, 4),
           "hlr_score": round(hlr_score, 4),
           "similarity_score": round(sim_score, 4),
           "variety_score": round(variety_score, 4),
           "final_score": round(final_score, 4)
})

    # Sort by final score descending
    ranked.sort(key=lambda x: x["final_score"], reverse=True)
    return ranked

