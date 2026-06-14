import json

with open("../data/questions.json", "r", encoding="utf-8") as f:
    questions = json.load(f)

# -----------------
#  QUESTION NODES
# -----------------

nodes = []

for q in questions:
    nodes.append({
        "id": q["title_slug"],
        "type": "question"
    })

# -----------------
#   TOPIC NODES
# -----------------

topics = {}

for q in questions:
    desc = q["description"].lower()

    if "array" in desc:
        topics["array"] = True

    if "string" in desc:
        topics["string"] = True

    if "linked list" in desc:
        topics["linked_list"] = True

    if "tree" in desc:
        topics["tree"] = True

    if "graph" in desc:
        topics["graph"] = True

for topic in topics:
    nodes.append({
        "id": topic,
        "type": "topic"
    })

# -----------------
#      EDGES
# -----------------

edges = []

for q in questions:
    desc = q["description"].lower()

    if "array" in desc:
        edges.append({
            "source": q["title_slug"],
            "target": "array",
            "relation": "has_topic"
        })

    if "string" in desc:
        edges.append({
            "source": q["title_slug"],
            "target": "string",
            "relation": "has_topic"
        })

    if "linked list" in desc:
        edges.append({
            "source": q["title_slug"],
            "target": "linked_list",
            "relation": "has_topic"
        })

    if "tree" in desc:
        edges.append({
            "source": q["title_slug"],
            "target": "tree",
            "relation": "has_topic"
        })

    if "graph" in desc:
        edges.append({
            "source": q["title_slug"],
            "target": "graph",
            "relation": "has_topic"
        })

# -----------------
#      SAVE
# -----------------

with open("../data/nodes.json", "w", encoding="utf-8") as f:
    json.dump(nodes, f, indent=2)

with open("../data/edges.json", "w", encoding="utf-8") as f:
    json.dump(edges, f, indent=2)

print("Total Nodes:", len(nodes))
print("Total Edges:", len(edges))