import logging
import os
import re
import psycopg2
from pathlib import Path
from psycopg2.extras import RealDictCursor
from dotenv import load_dotenv

log = logging.getLogger(__name__)

# Walk up from this file to find the repo root .env -- works regardless
# of what directory uvicorn is launched from.
_here = Path(__file__).resolve()
for _p in [_here.parent, *_here.parents]:
    if (_p / ".env").exists():
        load_dotenv(_p / ".env")
        break
else:
    load_dotenv()  # fallback

DATABASE_URL = os.environ.get("DATABASE_URL")


def get_connection():
    if not DATABASE_URL:
        raise RuntimeError("DATABASE_URL environment variable is not set")
    return psycopg2.connect(DATABASE_URL)


class _SQLAlchemyLikeSession:
    """
    Adapter so a raw psycopg2 connection can satisfy the SQLAlchemy
    Connection-style calling convention
    pipeline/recommender/services/user_graph_service.py::UserGraphService
    was written against:
        db.execute(sql_with_:named_params, params_dict).fetchone()/.fetchall()

    This repo has never had SQLAlchemy as a dependency (confirmed: no
    `create_engine`/`sessionmaker` anywhere) -- only psycopg2, which uses
    `%(name)s`-style bind params and returns a plain cursor, not a
    chainable result object. Rather than adding a whole ORM dependency
    just to satisfy one interface UserGraphService already expects,
    this translates `:name` -> `%(name)s` and returns the cursor itself
    (psycopg2 cursors already support .fetchone()/.fetchall() natively).

    Read-only by construction -- UserGraphService only ever SELECTs
    through `db` (see its module docstring: ML never writes to Postgres).
    """
    _PARAM_RE = re.compile(r":(\w+)\b")

    def __init__(self, conn):
        self._conn = conn

    def execute(self, sql, params=None):
        cur = self._conn.cursor()
        translated = self._PARAM_RE.sub(r"%(\1)s", sql)
        cur.execute(translated, params or {})
        return cur

    def close(self):
        self._conn.close()


def get_user_graph_session():
    """
    Returns a fresh `_SQLAlchemyLikeSession` wrapping a fresh psycopg2
    connection, for UserGraphService's `db=` param -- or None if
    DATABASE_URL isn't set or the connection fails. UserGraphService
    already treats db=None gracefully (falls back to a cold-start graph
    when the user lookup fails), same degrade-gracefully contract as
    db_env.py's neo4j_driver()/redis_client().

    Opens a FRESH connection per call (not a singleton/pool) -- matches
    every other function in this file (get_connection() is always called
    fresh, never cached), and psycopg2 connections aren't safe for
    concurrent use across threads, which FastAPI's sync `def` handlers can
    do. Caller is responsible for calling .close() on the returned session
    when done (see submission_controller.py/recommendation_controller.py).
    """
    if not DATABASE_URL:
        return None
    try:
        return _SQLAlchemyLikeSession(get_connection())
    except Exception as exc:
        log.warning("Postgres connection failed for UserGraphService read path: %s", exc)
        return None


def get_user_mastery(user_id: str) -> dict:
    conn = get_connection()
    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(
                "SELECT topic_id, mastery_score FROM user_topic_mastery WHERE user_id = %s ORDER BY updated_at DESC",
                (user_id,)
            )
            rows = cur.fetchall()
            return {
                row["topic_id"]: float(row["mastery_score"]) if row["mastery_score"] is not None else 0.0
                for row in rows
            }
    finally:
        conn.close()


def get_user_hlr(user_id: str) -> dict:
    conn = get_connection()
    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(
                "SELECT topic_id, half_life, last_review, p_recall, next_review_days FROM user_hlr_state WHERE user_id = %s",
                (user_id,)
            )
            rows = cur.fetchall()
            return {
                row["topic_id"]: {
                    "half_life": float(row["half_life"]) if row["half_life"] is not None else 1.0,
                    "last_review": str(row["last_review"]) if row["last_review"] is not None else None,
                    "p_recall": float(row["p_recall"]) if row["p_recall"] is not None else 0.5,
                    "next_review_days": float(row["next_review_days"]) if row["next_review_days"] is not None else 1.0
                }
                for row in rows
            }
    finally:
        conn.close()


def save_user_hlr(user_id: str, hlr_state: dict):
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            for topic_id, state in hlr_state.items():
                cur.execute("""
                    INSERT INTO user_hlr_state (user_id, topic_id, half_life, last_review, p_recall, next_review_days)
                    VALUES (%s, %s, %s, %s, %s, %s)
                    ON CONFLICT (user_id, topic_id)
                    DO UPDATE SET
                        half_life = EXCLUDED.half_life,
                        last_review = EXCLUDED.last_review,
                        p_recall = EXCLUDED.p_recall,
                        next_review_days = EXCLUDED.next_review_days
                """, (
                    user_id, topic_id,
                    state["half_life"], state.get("last_review"),
                    state.get("p_recall", 0.5), state.get("next_review_days", 1.0)
                ))
        conn.commit()
    finally:
        conn.close()


def save_user_mastery(user_id: str, mastery: dict):
    """
    Write BKT mastery scores to user_topic_mastery table.
    Used by seeding_controller.py when seeding initial mastery from
    LeetCode/Codeforces history. Uses ON CONFLICT DO NOTHING so it
    never overwrites mastery that was already computed from real
    in-platform submissions.
    """
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            for topic_id, mastery_score in mastery.items():
                cur.execute("""
                    INSERT INTO user_topic_mastery (user_id, topic_id, mastery_score, updated_at)
                    VALUES (%s, %s, %s, NOW())
                    ON CONFLICT (user_id, topic_id) DO NOTHING
                """, (user_id, topic_id, mastery_score))
        conn.commit()
    finally:
        conn.close()


def update_user_mastery(user_id: str, mastery: dict):
    """
    Upsert BKT mastery scores to user_topic_mastery table, overwriting any
    existing value. Used by StateUpdateService as a synchronous write-through
    on every submission so GET /mastery keeps working now that UserGraph
    (not Postgres) is the canonical store this value is computed from.
    Deliberately separate from save_user_mastery (seeding path), which must
    never overwrite real in-platform data with imported history.
    """
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            for topic_id, mastery_score in mastery.items():
                cur.execute("""
                    INSERT INTO user_topic_mastery (user_id, topic_id, mastery_score, updated_at)
                    VALUES (%s, %s, %s, NOW())
                    ON CONFLICT (user_id, topic_id) DO UPDATE SET
                        mastery_score = EXCLUDED.mastery_score,
                        updated_at = EXCLUDED.updated_at
                """, (user_id, topic_id, mastery_score))
        conn.commit()
    finally:
        conn.close()