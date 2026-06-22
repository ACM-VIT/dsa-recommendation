import pandas as pd

users = pd.DataFrame(columns=[
    "user_id",              # unique Knode user ID
    "cf_handle",            # Codeforces username
    "cf_rating",            # current CF rating
    "cf_max_rating",        # peak CF rating
    "lc_username",          # LeetCode username
    "lc_solved_easy",       # total easy problems solved on LC
    "lc_solved_medium",     # total medium problems solved on LC
    "lc_solved_hard",       # total hard problems solved on LC
    "lc_acceptance_rate",   # overall LC acceptance rate
])

problems = pd.DataFrame(columns=[
    "title_slug",           # unique problem ID e.g. "two-sum" — matches dataset
    "title",                # problem name e.g. "Two Sum"
    "description",          # full problem description
    "sample_test_cases",    # list of input/output test cases
    "difficulty_score",     # 1-5 — to be added by preprocessing team
    "source",               # leetcode (all problems in this dataset are LC)
])

topics = pd.DataFrame(columns=[
    "topic_slug",           # unique topic ID e.g. "sliding_window" — matches dataset
    "topic_name",           # display name e.g. "Sliding Window"
    "short_description",    # brief explanation of the topic
    "code_patterns",        # dict with python/cpp/java template code
    "is_root_topic",        # True if no prerequisites (e.g. arrays, strings)
    "difficulty_level",     # easy, medium, hard
])

patterns = pd.DataFrame(columns=[
    "pattern_id",           # e.g. "sliding_window"
    "name",                 # same as pattern_id
    "display_name",         # "Sliding Window"
    "description",          # brief explanation
    "topic_slug",           # which topic this pattern belongs to (FK to topics)
    "difficulty_tier",      # 1=easy, 2=medium, 3=hard within its topic
])

user_problem_edges = pd.DataFrame(columns=[
    "user_id",
    "problem_title_slug",   # matches title_slug in problem_nodes
    "verdict",              # OK, WA, TLE, RE
    "attempts",             # number of submissions before OK or gave up
    "time_consumed_millis", # actual time taken from CF submission
    "passed_test_count",    # how many test cases passed
    "memory_consumed_bytes",
    "timestamp",            # unix timestamp from CF
    "hints_taken",          # Knode only, null for CF data
    "source",               # "cf" or "knode"
])

user_topic_edges = pd.DataFrame(columns=[
    "user_id",
    "topic_slug",           # matches topic_slug in topic_nodes
    "mastery_score",        # 0 to 1 from BKT P(L)
    "solve_rate",           # OK count / total attempts
    "total_attempted",
    "total_solved",
    "last_attempted",       # timestamp
    "easiness_factor",      # SM2 easiness factor
    "sm2_interval",         # SM2 interval in days
    "sm2_repetition",       # SM2 repetition count
])

user_pattern_edges = pd.DataFrame(columns=[
    "user_id",
    "pattern_id",           # matches pattern_id in patterns
    "mastery_score",        # 0 to 100
    "attempt_count",
    "problems_solved",
    "last_attempted",
])

problem_topic_edges = pd.DataFrame(columns=[
    "source",               # title_slug of problem
    "target",               # topic_slug of topic
    "edgeType",             # "HAS_TOPIC"
    "is_primary_topic",     # True if main topic, False if secondary
])
topic_topic_edges = pd.DataFrame(columns=[
    "source",               # topic_slug
    "target",               # topic_slug
    "edgeType",             # "CO_OCCURS_WITH"
    "shared_problem_count", # how many problems have both topics
    "jaccard",              # jaccard similarity score between topics
])

topic_pattern_edges = pd.DataFrame(columns=[
    "topic_slug",           # matches topic_slug in topics
    "pattern_id",           # matches pattern_id in patterns
])

pattern_problem_edges = pd.DataFrame(columns=[
    "pattern_id",           # matches pattern_id in patterns
    "title_slug",           # matches title_slug in problems
])

schemas = {
    "NODE 1: Users": users,
    "NODE 2: Problems": problems,
    "NODE 3: Topics": topics,
    "NODE 4: Patterns": patterns,
    "EDGE 1: User -> Problem": user_problem_edges,
    "EDGE 2: User -> Topic mastery": user_topic_edges,
    "EDGE 3: User -> Pattern mastery": user_pattern_edges,
    "EDGE 4: Problem -> Topic (HAS_TOPIC)": problem_topic_edges,
    "EDGE 5: Topic -> Topic (CO_OCCURS_WITH)": topic_topic_edges,
    "EDGE 6: Topic -> Pattern": topic_pattern_edges,
    "EDGE 7: Pattern -> Problem": pattern_problem_edges,
}

for name, df in schemas.items():
    print("=" * 60)
    print(name)
    print("=" * 60)
    print(f"Columns ({len(df.columns)}):", list(df.columns))
    print()