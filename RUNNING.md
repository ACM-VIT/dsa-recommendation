# Running This Project

Quick reference for running everything in this repo. For Postman-specific walkthrough (scenarios, expected results), see [POSTMAN_GUIDE.md](POSTMAN_GUIDE.md).

## 1. Environment setup

Requires Python 3.12 and [uv](https://docs.astral.sh/uv/).

```bash
uv sync
```

Credentials come from a `.env` file at the repo root (see `db_env.py`'s module docstring for the full list of keys). At minimum for local dev you'll want:

```
QDRANT_URL=...
QDRANT_API_KEY=...
NEO4J_URI=...
NEO4J_PASSWORD=...
DATABASE_URL=postgresql://...
REDIS_URL=redis://...          # optional -- omit and the app degrades gracefully to Neo4j-only continuity
```

`REDIS_URL`/`DATABASE_URL` are read by `db_env.py`. `Neo4j`/`Redis`/Postgres are all optional at the code level (everything degrades to a cold-start graph if unreachable) but you'll lose cross-request state continuity without at least one of Redis or Neo4j configured and reachable.

**Windows + OneDrive-synced project folders**: `uv sync` defaults to hardlinking packages from its global cache into `.venv`, which fails against OneDrive's Files-On-Demand placeholders (`Access is denied` / `The cloud operation cannot be performed on a file with incompatible hardlinks`, os error 396). Fix it at the source instead of retrying:

```bash
uv sync --link-mode=copy
```

or set it once so every `uv` command in this repo uses it:

```
# .env, or your shell profile
UV_LINK_MODE=copy
```

If you're prompted "A virtual environment already exists... replace it?", answering **yes** deletes and rebuilds `.venv` from scratch — only do this if you actually need a clean venv; otherwise answer no and let `uv sync` update the existing one in place.

**IMPORTANT — commit your work.** This repo lives in a OneDrive-synced folder, and uncommitted changes have been lost twice in one session from branch switches / merges happening outside any single tool's visibility. Commit early and often; don't leave meaningful work uncommitted across a branch switch.

## 2. Run the API server

```bash
uvicorn main:app --reload --port 8000
```

- Swagger UI: `http://localhost:8000/docs`
- OpenAPI schema: `http://localhost:8000/openapi.json`

Every route has a typed response schema and description visible there. See [POSTMAN_GUIDE.md](POSTMAN_GUIDE.md) for importing `postman_collection.json` and a full request-by-request walkthrough with real expected results.

## 3. Run the test suite

```bash
uv run pytest tests/ -q
```

Takes ~3 minutes (~300 tests). Runs entirely offline — no real Qdrant/Neo4j/Postgres/Redis needed; everything is mocked. Run a single file with e.g. `uv run pytest tests/test_bkt.py -v`.

## 4. Run the CLI pipeline script (no server needed)

`run_full_pipeline.py` runs the whole thing end-to-end from one JSON request to one JSON response on stdout — useful for smoke-testing without spinning up uvicorn.

```bash
# recommend only, for an existing/cold user
python run_full_pipeline.py --input-json postman/pipeline_requests/recommend_only.json

# full loop: process a submission, THEN recommend
python run_full_pipeline.py --input-json postman/pipeline_requests/submission_then_recommend.json

# quick manual run via CLI flags instead of a JSON file
python run_full_pipeline.py some_user_id --k 5
```

stdout is always pure JSON (safe to pipe into `jq`/`python -m json.tool`); all progress/status logging goes to stderr.

## 5. Run the progression demo (no infra required)

Chains all 7 telemetry scenarios through BKT/HLR in a single process with an in-memory cache, so you can see the full bounded-delta / difficulty-credit / trivial-dampening behavior without needing Redis or Neo4j reachable:

```bash
python postman/demo_progression.py
```

This is also how the real numbers in [POSTMAN_GUIDE.md](POSTMAN_GUIDE.md)'s scenario table were generated.

## 6. Offline pipeline (ingestion / embeddings / graph training)

Only needed once, or when the problem catalog changes. `run_full_pipeline.py` auto-detects whether this has already run (checks Qdrant collection population) and runs it if needed:

```bash
python run_full_pipeline.py some_user_id --force-offline   # force re-run
python run_full_pipeline.py some_user_id --skip-offline    # never run, assume Qdrant is populated
```

Or run the offline stages directly if you need finer control — see `run_full_pipeline.py`'s `OFFLINE_STEPS` list for the exact commands (ingestion → embeddings → RGCN graph build + Qdrant ingest).

## Known environment caveats (not bugs, just things to know)

- **No connection pooling on Postgres.** Each request that needs `UserGraphService`'s Postgres bootstrap read opens and closes a fresh `psycopg2` connection (matches this codebase's existing pattern everywhere else). Fine for dev/moderate load; revisit with pgbouncer or similar before high-concurrency production use.
- **No distributed locking on concurrent submissions for the same user.** `POST /update`'s Redis/Neo4j read-modify-write cycle isn't atomic — two submissions for the same user_id landing at the exact same instant could race, with the second write overwriting the first's mastery update. Rare in practice (same user, same millisecond), but a real gap if you need hard guarantees under concurrent load from a single user.
- **Auth on `/seed_hlr`, `/seed_bkt` is a placeholder** (`routes/seeding.py::require_same_user`, documented in its own docstring) — reads an `X-User-Id` header with no signature/session verification. Any client can set that header to any value. Replace with real session/JWT verification before exposing these routes publicly.
- **`user_topic_mastery.topic_id` in Postgres stores a CUID**, not the flat topic slug (`"array"`, `"hash_map"`) the rest of the pipeline uses (Qdrant `topic_tags`, `data/problem_topic_edges_normalized.json`). Loading real Postgres mastery data doesn't currently connect to recommendations because of this taxonomy mismatch — flagged, not yet resolved (needs a decision on how to reconcile the two vocabularies, see the `topic` table's own `slug` column for a third, still-different naming convention).
