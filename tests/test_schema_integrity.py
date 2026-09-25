"""
Database-integrity tests for the Phase-2 schema hardening.

These tests exercise the REAL artifacts – schema.sql and
supabase/migrations/002_phase2_database_integrity.sql – against a live
PostgreSQL + pgvector instance, so the constraints the app depends on are
verified where they actually live (not just in a fake client):

  - fresh install: schema.sql applies and is idempotent
  - research_questions : status/priority enums, aspects NOT NULL,
                         answered ⇒ answered_at, embedding dims
  - research_sources   : per-session (question, canonical_url) uniqueness
                         for full-content rows, full+link coexistence,
                         unlinked rows exempt, source_type enum
  - posts              : status/platform/per-platform enums, completion
                         coherence (PUBLISHED ⇒ all targeted PUBLISHED,
                         PARTIAL_FAILURE ⇒ exactly one), NOT NULL timestamps
  - cleanup_stale_drafts(): staleness windows + idempotency
  - migration          : conflicting legacy data is repaired in place and
                         every constraint lands, with zero row loss in
                         research_questions / posts

Run against a disposable Postgres (e.g. docker run -p 5433:5432 ... pgvector/
pgvector:pg17). Skipped automatically when the DB is unreachable:

    TEST_DATABASE_URL=postgresql://postgres:postgres@localhost:5433/personapulse_test \
        .venv/bin/python -m unittest tests.test_schema_integrity -v
"""

from __future__ import annotations

import os
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

import psycopg
from psycopg import errors as pg_errors

_REPO_ROOT = Path(__file__).resolve().parents[1]
_SCHEMA_SQL = (_REPO_ROOT / "schema.sql").read_text()
_MIGRATION_SQL = (_REPO_ROOT / "supabase/migrations/002_phase2_database_integrity.sql").read_text()
_DATABASE_URL = os.environ.get(
    "TEST_DATABASE_URL",
    "postgresql://postgres:postgres@localhost:5433/personapulse_test",
)


def _apply(conn, sql: str) -> None:
    conn.execute(sql)


def _reset(conn) -> None:
    """Start from a truly empty public schema in a disposable test DB."""
    _apply(conn, "DROP SCHEMA IF EXISTS public CASCADE; CREATE SCHEMA public;")
    _apply(conn, "CREATE EXTENSION IF NOT EXISTS vector;")
    conn.execute("SET search_path TO public")


def _dim_vector(n: int, value: float = 0.5) -> str:
    return "[" + ",".join([repr(value)] * n) + "]"


def _try_connect():
    try:
        conn = psycopg.connect(_DATABASE_URL, autocommit=True)
        _reset(conn)
        return conn
    except Exception:  # pragma: no cover - environment-dependent
        return None


@unittest.skipUnless(_try_connect() is not None, "PostgreSQL + pgvector unavailable (TEST_DATABASE_URL)")
class SchemaIntegrityTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.conn = _try_connect()

    def setUp(self):
        _reset(type(self).conn)

    def _fresh(self):
        """Apply the hardened full schema (all non-migration tests)."""
        _apply(type(self).conn, _SCHEMA_SQL)

    def _insert(self, sql: str, params=()):
        return type(self).conn.execute(sql, params)

    def _insert_question(self, **overrides):
        row = {
            "topic": "Agentic AI",
            "question": "How do agentic systems coordinate?",
            "aspects": '["memory", "tool use"]',
            "status": "proposed",
            "priority": "normal",
            "answered_at": None,
        }
        row.update(overrides)
        cols = ", ".join(row)
        vals = ", ".join(["%s"] * len(row))
        return self._insert(
            f"INSERT INTO research_questions ({cols}) VALUES ({vals}) RETURNING id",
            tuple(row.values()),
        ).fetchone()[0]

    def _insert_post(self, **overrides):
        now = datetime.now(timezone.utc)
        row = {
            "platform": "both",
            "topic": "The Rise of Agentic AI",
            "content": "LINKEDIN:\n...\n\nX:\n...",
            "status": "PENDING",
            "linkedin_status": "PENDING",
            "x_status": "PENDING",
            "created_at": now,
            "updated_at": now,
        }
        row.update(overrides)
        cols = ", ".join(row)
        vals = ", ".join(["%s"] * len(row))
        return self._insert(
            f"INSERT INTO posts ({cols}) VALUES ({vals}) RETURNING id",
            tuple(row.values()),
        ).fetchone()[0]

    def _insert_source(self, question_id, url, body, source_type="secondary", **overrides):
        row = {
            "research_question_id": question_id,
            "url": url,
            "canonical_url": "https://ex.com/a",  # fingerprint written by the app
            "title": "T",
            "body": body,
            "source_type": source_type,
        }
        row.update(overrides)
        cols = ", ".join(row)
        vals = ", ".join(["%s"] * len(row))
        return self._insert(
            f"INSERT INTO research_sources ({cols}) VALUES ({vals}) RETURNING id",
            tuple(row.values()),
        )

    # ── fresh install + idempotency ──────────────────────────────────────

    def test_schema_applies_to_a_fresh_database_and_is_idempotent(self):
        self._fresh()
        _apply(type(self).conn, _SCHEMA_SQL)  # second run must be a no-op

        constraint_names = {
            r[0] for r in self._insert(
                "SELECT conname FROM pg_constraint WHERE conname LIKE %s", ("%_check",)
            ).fetchall()
        }
        for expected in (
            "posts_platform_check",
            "posts_status_check",
            "posts_linkedin_status_check",
            "posts_x_status_check",
            "posts_completion_check",
            "research_questions_status_check",
            "research_questions_priority_check",
            "research_questions_answered_at_check",
            "research_questions_embedding_dims_check",
            "research_sources_source_type_check",
        ):
            self.assertIn(expected, constraint_names, expected)

        index_names = {r[0] for r in self._insert(
            "SELECT indexname FROM pg_indexes WHERE schemaname = %s", ("public",)
        ).fetchall()}
        self.assertIn("research_sources_session_url_uniq", index_names)
        self.assertIn("posts_status_created_at_idx", index_names)
        self.assertIn("posts_status_updated_at_idx", index_names)
        # the bare single-column status/created_at/updated_at indexes are gone
        for removed in ("posts_status_idx", "posts_created_at_idx", "posts_updated_at_idx"):
            self.assertNotIn(removed, index_names)

    # ── research_questions integrity ─────────────────────────────────────

    def test_research_questions_reject_unknown_status_and_priority(self):
        self._fresh()
        with self.assertRaises(pg_errors.CheckViolation):
            self._insert_question(status="publish_ready")
        with self.assertRaises(pg_errors.CheckViolation):
            self._insert_question(priority="urgent")
        self.assertIsNotNone(self._insert_question(status="researching"))

    def test_research_questions_enforce_answered_at_bidirectional_invariant(self):
        self._fresh()
        # answered must carry its timestamp …
        with self.assertRaises(pg_errors.CheckViolation):
            self._insert_question(status="answered", answered_at=None)
        # … and no OTHER status may carry one
        with self.assertRaises(pg_errors.CheckViolation):
            self._insert_question(status="researching", answered_at=datetime(2026, 9, 26, tzinfo=timezone.utc))
        with self.assertRaises(pg_errors.CheckViolation):
            self._insert_question(status="dropped", answered_at=datetime(2026, 9, 26, tzinfo=timezone.utc))
        with self.assertRaises(pg_errors.NotNullViolation):
            self._insert(
                "INSERT INTO research_questions (topic, question, aspects, status) "
                "VALUES (%s, %s, NULL, 'proposed')",
                ("T", "Q"),
            )
        qid = self._insert_question(
            status="answered", answered_at=datetime(2026, 9, 26, tzinfo=timezone.utc)
        )
        self.assertIsNotNone(qid)

    def test_research_questions_enforce_embedding_dimensions(self):
        self._fresh()
        with self.assertRaises(Exception):
            self._insert_question(embedding="[1.0,2.0,3.0]")
        qid = self._insert_question(embedding=_dim_vector(768))
        kept = self._insert(
            "SELECT embedding IS NOT NULL FROM research_questions WHERE id = %s", (qid,)
        ).fetchone()[0]
        self.assertTrue(kept)

    # ── research_sources integrity ───────────────────────────────────────

    def test_one_full_source_per_session_and_canonical_url(self):
        self._fresh()
        q1 = self._insert_question()
        q2 = self._insert_question()
        self._insert_source(q1, "https://ex.com/a", "full A",
                            canonical_url="https://ex.com/a")
        self._insert_source(q2, "https://ex.com/a", "full A again",
                            canonical_url="https://ex.com/a")
        # duplicate full row in the SAME session is rejected …
        with self.assertRaises(pg_errors.UniqueViolation):
            self._insert_source(q1, "https://ex.com/a", "dup", canonical_url="https://ex.com/a")
        # … even under a trivially different spelling of the same URL
        with self.assertRaises(pg_errors.UniqueViolation):
            self._insert_source(q1, "https://www.ex.com/a/?utm_source=rss#frag", "dup",
                                canonical_url="https://ex.com/a")
        # a per-session link row (empty body) may coexist with the full record
        self._insert_source(q1, "https://ex.com/a", "", canonical_url="https://ex.com/a")
        # unlinked rows (research_question_id NULL) are exempt
        self._insert_source(None, "https://ex.com/a", "orphan 1", canonical_url="https://ex.com/a")
        self._insert_source(None, "https://ex.com/a", "orphan 2", canonical_url="https://ex.com/a")

    def test_research_sources_reject_unknown_source_type(self):
        self._fresh()
        qid = self._insert_question()
        with self.assertRaises(pg_errors.CheckViolation):
            self._insert_source(qid, "https://ex.com/x", "body", source_type="tertiary")

    # ── posts integrity ──────────────────────────────────────────────────

    def test_posts_reject_unknown_status_and_platform(self):
        self._fresh()
        with self.assertRaises(pg_errors.CheckViolation):
            self._insert_post(platform="TikTok")
        with self.assertRaises(pg_errors.CheckViolation):
            self._insert_post(status="DONE")
        with self.assertRaises(pg_errors.CheckViolation):
            self._insert_post(linkedin_status="PARTIAL_FAILURE")
        self.assertIsNotNone(self._insert_post(linkedin_status="PUBLISHING"))

    def test_posts_completion_coherence(self):
        self._fresh()
        # — platform=both: PUBLISHED requires both platforms …
        self._insert_post(platform="both", status="PUBLISHED",
                          linkedin_status="PUBLISHED", x_status="PUBLISHED")
        with self.assertRaises(pg_errors.CheckViolation):
            self._insert_post(platform="both", status="PUBLISHED",
                              linkedin_status="PUBLISHED", x_status="PENDING")
        # … and PARTIAL_FAILURE requires exactly one successful platform
        self._insert_post(platform="both", status="PARTIAL_FAILURE",
                          linkedin_status="PUBLISHED", x_status="FAILED")
        with self.assertRaises(pg_errors.CheckViolation):
            self._insert_post(platform="both", status="PARTIAL_FAILURE",
                              linkedin_status="PUBLISHED", x_status="PUBLISHED")
        with self.assertRaises(pg_errors.CheckViolation):
            self._insert_post(platform="both", status="PARTIAL_FAILURE",
                              linkedin_status="FAILED", x_status="FAILED")

        # — platform=linkedin: completion depends only on linkedin_status
        self._insert_post(platform="linkedin", status="PUBLISHED",
                          linkedin_status="PUBLISHED", x_status="FAILED")
        with self.assertRaises(pg_errors.CheckViolation):
            # an unrelated x success must NOT make the post appear published
            self._insert_post(platform="linkedin", status="PUBLISHED",
                              linkedin_status="FAILED", x_status="PUBLISHED")
        self._insert_post(platform="linkedin", status="PARTIAL_FAILURE",
                          linkedin_status="PUBLISHED", x_status="FAILED")
        with self.assertRaises(pg_errors.CheckViolation):
            # … and must NOT make it appear partially failed either
            self._insert_post(platform="linkedin", status="PARTIAL_FAILURE",
                              linkedin_status="FAILED", x_status="PUBLISHED")

        # — platform=x: completion depends only on x_status
        self._insert_post(platform="x", status="PUBLISHED",
                          x_status="PUBLISHED", linkedin_status="FAILED")
        with self.assertRaises(pg_errors.CheckViolation):
            self._insert_post(platform="x", status="PUBLISHED",
                              x_status="PENDING", linkedin_status="PUBLISHED")
        self._insert_post(platform="x", status="PARTIAL_FAILURE",
                          x_status="PUBLISHED", linkedin_status="FAILED")
        with self.assertRaises(pg_errors.CheckViolation):
            self._insert_post(platform="x", status="PARTIAL_FAILURE",
                              x_status="FAILED", linkedin_status="PUBLISHED")

        # mid-flight PUBLISHING with mixed per-platform states stays allowed
        self._insert_post(status="PUBLISHING", linkedin_status="PUBLISHED", x_status="PUBLISHING")

    def test_posts_require_timestamps(self):
        self._fresh()
        with self.assertRaises(pg_errors.NotNullViolation):
            self._insert_post(created_at=None)

    # ── cleanup_stale_drafts ─────────────────────────────────────────────

    def test_cleanup_stale_drafts_applies_staleness_windows(self):
        self._fresh()
        now = datetime.now(timezone.utc)
        old_id = self._insert_post(
            status="PENDING", created_at=now - timedelta(hours=25),
            updated_at=now - timedelta(hours=25),
        )
        partial_id = self._insert_post(
            status="PARTIAL_FAILURE", created_at=now - timedelta(hours=25),
            updated_at=now - timedelta(hours=25),
            linkedin_status="PUBLISHED", x_status="FAILED",
        )
        dead_claim_id = self._insert_post(
            status="PUBLISHING", created_at=now - timedelta(hours=2),
            updated_at=now - timedelta(minutes=16),
        )
        fresh_id = self._insert_post(status="PENDING")
        live_claim_id = self._insert_post(status="PUBLISHING")

        self._insert("SELECT cleanup_stale_drafts()")

        def status(post_id):
            return self._insert(
                "SELECT status FROM posts WHERE id = %s", (post_id,)
            ).fetchone()[0]

        self.assertEqual(status(old_id), "REJECTED")
        self.assertEqual(status(partial_id), "REJECTED")
        self.assertEqual(status(dead_claim_id), "PENDING")
        self.assertEqual(status(fresh_id), "PENDING")
        self.assertEqual(status(live_claim_id), "PUBLISHING")

        # idempotent: a second run must not mutate anything further
        before = self._insert("SELECT status FROM posts ORDER BY id").fetchall()
        self._insert("SELECT cleanup_stale_drafts()")
        after = self._insert("SELECT status FROM posts ORDER BY id").fetchall()
        self.assertEqual(before, after)

    # ── migration on legacy data ─────────────────────────────────────────

    def test_migration_repairs_legacy_data_without_losing_rows(self):
        conn = type(self).conn
        _apply(conn, """
            CREATE TABLE posts (
                id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                platform        VARCHAR(32) NOT NULL,
                topic           VARCHAR(256) NOT NULL,
                content         TEXT NOT NULL,
                article_url     TEXT,
                image_url       TEXT,
                embedding       vector,
                status          VARCHAR(32) DEFAULT 'PENDING',
                linkedin_status VARCHAR(32) DEFAULT 'PENDING',
                x_status        VARCHAR(32) DEFAULT 'PENDING',
                created_at      TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
                updated_at      TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
            );
            CREATE TABLE research_questions (
                id          UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                topic       TEXT NOT NULL,
                question    TEXT NOT NULL,
                aspects     JSONB DEFAULT '[]',
                status      VARCHAR(32) DEFAULT 'proposed',
                priority    VARCHAR(16) DEFAULT 'normal',
                embedding   vector,
                created_at  TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
            );
            CREATE TABLE research_sources (
                id                   UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                research_question_id UUID REFERENCES research_questions(id) ON DELETE CASCADE,
                url                  TEXT NOT NULL,
                title                TEXT DEFAULT '',
                body                 TEXT DEFAULT '',
                source               TEXT DEFAULT '',
                published            TEXT DEFAULT '',
                score                DOUBLE PRECISION DEFAULT 0,
                source_type          VARCHAR(32) DEFAULT 'secondary',
                accessed_at          TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
            );
        """)

        q_answered = self._insert(
            "INSERT INTO research_questions (topic, question, status) "
            "VALUES ('T', 'answered-legacy', 'answered') RETURNING id"
        ).fetchone()[0]
        q_bad_status = self._insert(
            "INSERT INTO research_questions (topic, question, status, priority, aspects) "
            "VALUES ('T', 'bad-status', 'publish_ready', 'urgent', NULL) RETURNING id"
        ).fetchone()[0]
        q_bad_dims = self._insert(
            "INSERT INTO research_questions (topic, question, status, embedding) "
            "VALUES ('T', 'bad-dims', 'proposed', %s) RETURNING id",
            ("[1.0,2.0,3.0]",),
        ).fetchone()[0]

        session = q_answered
        self._insert(
            "INSERT INTO research_sources (research_question_id, url, body, accessed_at) "
            "VALUES (%s, 'https://ex.com/a', 'Content A1', '2026-09-20T08:00:00Z')", (session,)
        )
        self._insert(
            "INSERT INTO research_sources (research_question_id, url, body, accessed_at) "
            "VALUES (%s, 'https://www.ex.com/a/?utm_source=rss#frag', 'Content A2', '2026-09-21T08:00:00Z')",
            (session,),
        )
        # full + link row for the same canonical URL in one session
        self._insert(
            "INSERT INTO research_sources (research_question_id, url, body) "
            "VALUES (%s, 'https://ex.com/b', 'Full B')", (session,)
        )
        self._insert(
            "INSERT INTO research_sources (research_question_id, url, title) "
            "VALUES (%s, 'https://ex.com/b', 'Link B')", (session,)
        )
        # unlinked duplicates (one carries an out-of-enum source_type)
        self._insert(
            "INSERT INTO research_sources (research_question_id, url, body, source_type) "
            "VALUES (NULL, 'https://ex.com/c', 'Orphan 1', 'tertiary')"
        )
        self._insert(
            "INSERT INTO research_sources (research_question_id, url, body) "
            "VALUES (NULL, 'https://ex.com/c', 'Orphan 2')"
        )

        # legacy posts (only possible while the constraints do not exist yet)
        self._insert(
            "INSERT INTO posts (platform, topic, content, status) "
            "VALUES ('both', 'legacy-published', 'c', 'PUBLISHED')"
        )
        self._insert(
            "INSERT INTO posts (platform, topic, content, status, linkedin_status, x_status) "
            "VALUES ('both', 'legacy-weird', 'c', 'weird', 'PENDING', 'PENDING')"
        )
        self._insert(
            "INSERT INTO posts (platform, topic, content, status, linkedin_status) "
            "VALUES ('LinkedIn', 'legacy-pub-fail', 'c', 'PUBLISHED', 'FAILED')"
        )

        posts_before = self._insert("SELECT COUNT(*) FROM posts").fetchone()[0]
        questions_before = self._insert("SELECT COUNT(*) FROM research_questions").fetchone()[0]

        # Snapshot every legacy (session, raw url, body) BEFORE the migration so
        # the reference-preservation invariant can be proven afterwards: the set
        # of distinct canonical fingerprints must be identical before and after,
        # and every fingerprint that carried full content must still carry it.
        self._insert("CREATE TEMP TABLE legacy_sources AS SELECT * FROM research_sources")

        _apply(conn, _MIGRATION_SQL)

        # canonical_url backfilled + NOT NULL for every row
        self.assertEqual(
            self._insert(
                "SELECT COUNT(*) FROM research_sources WHERE canonical_url IS NULL"
            ).fetchone()[0],
            0,
        )
        # NO valid research session loses its canonical source reference: the
        # distinct (session, canonical_url) fingerprints are unchanged.
        legacy_fingerprints = set(self._insert(
            "SELECT DISTINCT research_question_id, canonicalize_source_url(url) "
            "FROM legacy_sources"
        ).fetchall())
        surviving_fingerprints = set(self._insert(
            "SELECT DISTINCT research_question_id, canonical_url FROM research_sources"
        ).fetchall())
        self.assertEqual(surviving_fingerprints, legacy_fingerprints)
        # … and every fingerprint that once carried full content still does.
        legacy_full = set(self._insert(
            "SELECT DISTINCT research_question_id, canonicalize_source_url(url) "
            "FROM legacy_sources WHERE body <> ''"
        ).fetchall())
        surviving_full = set(self._insert(
            "SELECT DISTINCT research_question_id, canonical_url FROM research_sources "
            "WHERE body <> ''"
        ).fetchall())
        self.assertLessEqual(legacy_full, surviving_full)

        # duplicates merged per session, keeping the RICHEST row
        rows_for = self._insert(
            "SELECT canonical_url, body FROM research_sources "
            "WHERE research_question_id = %s ORDER BY canonical_url", (session,)
        ).fetchall()
        canonical_counts = {}
        for canon, body in rows_for:
            canonical_counts[canon] = canonical_counts.get(canon, 0) + 1
        self.assertEqual(canonical_counts["https://ex.com/a"], 1)
        self.assertEqual(canonical_counts["https://ex.com/b"], 1)
        self.assertTrue(any(c == "https://ex.com/b" and b == "Full B" for c, b in rows_for))
        # for /a both legacy copies carried content: the newest-accessed survives
        bodies_a = [b for c, b in rows_for if c == "https://ex.com/a"]
        self.assertEqual(bodies_a, ["Content A2"])
        self.assertEqual(
            self._insert(
                "SELECT COUNT(*) FROM research_sources WHERE research_question_id IS NULL"
            ).fetchone()[0],
            1,
        )

        # legacy rows cleaned up in place, NOTHING deleted from those tables
        self.assertEqual(
            self._insert("SELECT COUNT(*) FROM posts").fetchone()[0], posts_before
        )
        self.assertEqual(
            self._insert("SELECT COUNT(*) FROM research_questions").fetchone()[0], questions_before
        )
        self.assertEqual(
            self._insert(
                "SELECT status, priority, aspects FROM research_questions WHERE id = %s",
                (q_bad_status,),
            ).fetchone(),
            ("proposed", "normal", []),
        )
        self.assertTrue(self._insert(
            "SELECT answered_at IS NOT NULL FROM research_questions WHERE id = %s",
            (q_answered,),
        ).fetchone()[0])
        # non-answered sessions carry NO answered_at after the migration
        self.assertFalse(self._insert(
            "SELECT answered_at IS NOT NULL FROM research_questions WHERE id = %s",
            (q_bad_status,),
        ).fetchone()[0])
        self.assertFalse(self._insert(
            "SELECT embedding IS NOT NULL FROM research_questions WHERE id = %s",
            (q_bad_dims,),
        ).fetchone()[0])

        # legacy posts: per-platform mirror backfilled, unknown values normalized
        self.assertEqual(
            self._insert(
                "SELECT linkedin_status, x_status FROM posts WHERE topic = 'legacy-published'"
            ).fetchone(),
            ("PUBLISHED", "PUBLISHED"),
        )
        self.assertEqual(
            self._insert(
                "SELECT status, platform FROM posts WHERE topic = 'legacy-weird'"
            ).fetchone(),
            ("PENDING", "both"),
        )
        # one durably FAILED platform while the other is an unconfirmed default →
        # the top-level PUBLISHED legacy claim is normalized to PENDING
        # (the app's deriveFinalStatus(): no platform confirmed ⇒ PENDING)
        self.assertEqual(
            self._insert(
                "SELECT status, linkedin_status, x_status FROM posts "
                "WHERE topic = 'legacy-pub-fail'"
            ).fetchone(),
            ("PENDING", "FAILED", "PENDING"),
        )

        # re-run to be sure the migration is idempotent even against clean data
        _apply(conn, _MIGRATION_SQL)

        # every constraint is live after the migration
        names = {r[0] for r in self._insert(
            "SELECT conname FROM pg_constraint WHERE conname LIKE %s", ("%_check",)
        ).fetchall()}
        for expected in (
            "posts_status_check",
            "posts_completion_check",
            "research_questions_status_check",
            "research_questions_answered_at_check",
            "research_sources_source_type_check",
        ):
            self.assertIn(expected, names)


if __name__ == "__main__":
    unittest.main()