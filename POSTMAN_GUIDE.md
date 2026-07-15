# Postman Guide: DSA Recommendation Service

This walks through testing every endpoint with real, captured expected results — not fabricated ones. Everything below was actually run against this codebase.

## 1. Start the server

```bash
uvicorn main:app --reload --port 8000
```

Swagger docs live at `http://localhost:8000/docs` once it's up — every route now shows a typed response schema there too.

## 2. Import into Postman

1. **File → Import** → select `postman_collection.json` from the repo root.
2. You'll see six folders: `Root`, `Telemetry Scenarios`, `Submission`, `Recommendation`, `Mastery`, `Seeding`, plus a documentation-only `Full Pipeline` folder (that one isn't HTTP — see §6).
3. Click the collection name → **Variables** tab. Confirm:
   - `base_url` = `http://localhost:8000` (change if you're not running locally)
   - `user_id` = whatever default you want for the `Submission`/`Recommendation`/`Mastery`/`Seeding` folders (the `Telemetry Scenarios` folder has its own userId baked into each request body instead, so it's not affected by this variable).

Alternative: **File → Import → Link** → `http://localhost:8000/openapi.json` gives you a collection auto-generated from the live schema, always in sync.

## 3. Telemetry Scenarios — run these first, in order

Each request in this folder is `POST /update` with a body from `postman/telemetry_samples/*.json`. Run them top to bottom once each (5a before 5b matters — see below).

| # | Request | What it proves | `updatedMastery` (array topic, 0.15 → ?) | Real `observed_score` |
|---|---|---|---|---|
| 1 | Cold start, easy solve | difficulty=0.15 (low) → the CAP ITSELF scales down (not just the raw signal), so this lands below the neutral +0.12, not clipped to it | **0.15 → 0.2448** (Δ = **+0.0948**) | 0.7742 |
| 2 | Struggling, many attempts | 8 submissions + 4 hints drags the confidence-adjusted signal down even though verdict=OK | 0.15 → **0.2361** (Δ = +0.0861) | 0.3648 |
| 3 | Wrong answer | Non-OK verdict caps the signal at ≤0.35, no BKT learning-transition credit | 0.15 → **0.2273** (Δ = +0.0773, from Bayesian blending only) | 0.35 |
| 4 | Moderate-difficulty clean solve | difficulty=0.5 is the cap-scaling formula's neutral point → reproduces the un-scaled +0.12 exactly | 0.15 → **0.27** (Δ = +0.12) | 0.96 |
| 5a | Warm-up before trivial | difficulty=0.5 (neutral), raises `array` mastery so 5b has something to be trivial *relative to* | 0.15 → **0.27** (Δ = +0.12) | 0.96 |
| 5b | Trivial-difficulty solve | Same user, `array` at difficulty=0.05 (well below their now-0.27 mastery): **both** mechanisms stack — the low absolute difficulty shrinks the cap, AND the large relative gap below current mastery dampens further | 0.27 → **0.3383** (Δ = **+0.0683**) — compare to `hash_map`/`simulation` in the *same* response, which only got the cap-scaling (no prior mastery → no relative-gap dampening), landing at Δ = +0.0876 | 0.73 |
| 6 | Hard stretch solve | difficulty=0.9 (high) → the cap itself scales UP to 1.3× — the largest delta of any scenario here, genuinely more than the neutral +0.12 | 0.15 → **0.2988** (Δ = **+0.1488**, the largest of all six scenarios) | **1.0** (max) |

**Difficulty now differentiates mastery gain in two stacking ways** (`pipeline/recommender/bkt.py`):
1. **Cap-scaling by absolute difficulty** — the per-submission ceiling itself is `MAX_MASTERY_DELTA × [0.7..1.3]` based on the problem's difficulty (0.5 = neutral, matches `difficulty=None`). This is what makes scenarios 1 and 6 land at different deltas (+0.0948 vs +0.1488) instead of both clipping to the same flat +0.12 — a real bug found during testing, since a cold-start user's raw Bayesian delta is almost always large enough to saturate any flat cap regardless of difficulty, silently erasing the difficulty signal.
2. **Trivial-gap dampening by relative difficulty-vs-mastery** — on top of #1, a solve well below the user's *own current mastery* on that topic (scenario 5b's `array`) gets an additional multiplicative dampening, distinct from the topic's absolute difficulty.

## 4. Recommendation, Mastery, Urgency

Run `GET /recommend/{{user_id}}`, `GET /mastery/{{user_id}}`, `GET /urgency/{{user_id}}` with `user_id` set to e.g. `postman_demo_user_1`.

**Important — read this before you conclude anything looks broken:**

- **`GET /mastery` and `GET /urgency` will show empty (`{}`) for any demo/test `user_id` that doesn't already exist as a real row in the backend's Postgres `User` table.** This is not a sign that mastery tracking failed. `POST /update` writes to the canonical UserGraph (Redis/Neo4j) correctly regardless — confirmed above, mastery genuinely moved. But these two *specific* read endpoints are legacy, Postgres-only projections (`database/postgres/db.py::get_user_mastery`/`get_user_hlr`), and the write-through to Postgres is a foreign-key-constrained insert that silently fails (logged, not raised) for a `user_id` with no real `User` row. **To see real data via these two endpoints, use a `user_id` that's already provisioned in your Postgres `User` table.**
- **`GET /recommend/{user_id}` may return fewer results than `limit`, or zero, for a user immediately after their very first submission.** This was traced to the currently-ingested Qdrant catalog having **zero** `array`-tagged problems below difficulty 0.34 (checked directly: 0 easy / 646 medium / 144 hard) — combined with a fresh user's low mastery (~0.27) and the ZPD success-probability filter (keeps only candidates predicted 55-80% likely to succeed), almost nothing clears the bar. This is a **data/offline-ingestion characteristic of this environment's Qdrant instance**, not a bug in the online recommender — a genuinely cold user (zero submissions ever) actually gets recommendations fine (confirmed: 3 results, `course_path` sourced), because their missing-mastery case defaults to a neutral score rather than a computed one. If your Qdrant is populated with a fuller catalog, this won't be visible.
- Real captured example for a genuinely cold user (`truly_never_touched_user_2`):
  ```json
  {
    "user_id": "truly_never_touched_user_2",
    "recommendations": [
      {"problem_id": "bxmw7pseor0fh8zr8tn1xqgv", "title": "Make Array Strictly Increasing", "difficulty_score": 0.421, "source": "course_path", ...},
      {"problem_id": "rqkcqezk0l63f87059u790ta", "title": "Employee Importance", "difficulty_score": 0.736, "source": "course_path", ...},
      {"problem_id": "bf39aap4c917x49myqxj5kn5", "title": "Maximum 69 Number", "difficulty_score": 0.498, "source": "course_path", ...}
    ]
  }
  ```

## 5. Validation and error cases (all confirmed working)

| Request | Expected | Confirmed |
|---|---|---|
| `GET /recommend/{user_id}?limit=999` | `422`, `"Input should be less than or equal to 50"` | ✅ |
| `GET /recommend/{user_id}?limit=0` | `422`, `"Input should be greater than or equal to 1"` | ✅ |
| `POST /update` with `verdict` missing | `422`, Pydantic "Field required" | ✅ |
| `POST /update` with `normalisedScore` or `problemDifficulty` outside `[0,1]` | `422`, Pydantic range error | ✅ |
| `POST /seed_hlr/{user_id}` with `X-User-Id` header not matching the path | `403 Cannot seed data for another user` | ✅ |
| `POST /seed_hlr/{user_id}` with no `X-User-Id` header | `401 Not authenticated` | ✅ |
| `POST /seed_hlr/{user_id}` for a user not in Postgres | `200 {"message": "User not found"}` | ✅ |

**Design note on `limit`:** the original code silently clamped out-of-range `limit` values instead of rejecting them. Adding `Query(ge=1, le=50)` made this stricter (422 instead of a silent clamp-and-200) — a deliberate, confirmed trade-off, not an oversight. Same reasoning applies to `normalisedScore`/`problemDifficulty`'s `[0,1]` bounds — a malformed value used to be silently clamped deep inside telemetry.py and absorbed into a real mastery change instead of surfaced as an error.

## 6. `run_full_pipeline.py` — the full submission → state-update → recommend loop in one script

This CLI script now accepts an optional `"submission"` key, routing it through the same `StateUpdateService` `POST /update` uses, before generating recommendations:

```bash
python run_full_pipeline.py --input-json postman/pipeline_requests/submission_then_recommend.json
python run_full_pipeline.py --input-json postman/pipeline_requests/recommend_only.json   # no submission, recommend only
```

stdout is *always* pure JSON (a real bug was found and fixed here: `bkt.py`/`hlr.py` had `print()` statements landing on stdout and corrupting this script's output — they now go to stderr). The response includes a `state_update` key (same shape as `POST /update`'s response) when a submission was processed.

## 7. `postman/demo_progression.py` — guaranteed state continuity, no infra required

Postman calls to a running server only persist state across separate requests if **Neo4j is reachable**, or **Redis** is configured (`REDIS_URL` in `.env` — now wired into both controllers). In a sandboxed/offline environment with neither reachable, every `POST /update` may start from a fresh graph.

`postman/demo_progression.py` sidesteps this by chaining all 7 scenarios through one shared in-memory cache within a single Python process, so the mastery-continuity effects (especially scenario 5's dampening) are reliably observable regardless of infrastructure:

```bash
python postman/demo_progression.py
```

Its output is the real source for the numbers in the table in §3.

## 8. Files reference

```
postman_collection.json                          # import this into Postman
postman/telemetry_samples/*.json                  # the 7 request bodies, individually
postman/pipeline_requests/*.json                  # run_full_pipeline.py request examples
postman/demo_progression.py                        # chained, infra-independent demo + expected-results generator
postman/expected_responses/captured_run_2026-07-15.json  # real captured responses for every endpoint (status + body), from the run this guide's numbers came from
```
