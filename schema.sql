-- ============================================================
-- PersonaPulse AI Social Media Agent - Database Schema
-- ============================================================
-- Run this against your Supabase PostgreSQL instance.
-- Requires the pgvector extension to be enabled.
--
-- This file is IDEMPOTENT and safe to run repeatedly, on a fresh database
-- AND on an existing install. Tables are created with the hardened column
-- layout, then a guarded migration stage:
--   1. repairs legacy rows that predate the integrity constraints
--      (backfills / normalizes in place – never deletes rows), and
--   2. adds the DB-enforced integrity constraints via DO-block guards so
--      re-running the file is always a no-op once they exist.
--
-- The same migration stage ships standalone for teams that deploy through
-- Supabase migrations: supabase/migrations/002_phase2_database_integrity.sql

-- Enable pgvector extension for semantic similarity search
CREATE EXTENSION IF NOT EXISTS vector;

-- ============================================================
-- TABLE 1: posts (Episodic Memory)
-- Stores all published posts, pending drafts, and historical
-- examples used for style learning and deduplication.
--
-- State machine (enforced by DB CHECKs below):
--   top-level   : PENDING | PUBLISHING | PUBLISHED | PARTIAL_FAILURE | REJECTED
--   per-platform: PENDING | PUBLISHING | PUBLISHED | FAILED
-- The webhook claims a row atomically (PENDING/PARTIAL_FAILURE → PUBLISHING,
-- with RETURNING) and writes per-platform PUBLISHING as a write-ahead marker
-- before each external publish call, so a crash never loses an in-flight
-- attempt. Completion invariants: a PUBLISHED row has every targeted
-- platform PUBLISHED; a PARTIAL_FAILURE row has exactly one platform
-- PUBLISHED (and at least one that failed).
-- ============================================================
CREATE TABLE IF NOT EXISTS posts (
    id          UUID          PRIMARY KEY DEFAULT gen_random_uuid(),
    platform    VARCHAR(32)   NOT NULL,                                -- 'linkedin', 'x', or 'both'
    topic       VARCHAR(256)  NOT NULL,
    content     TEXT          NOT NULL,
    article_url TEXT,
    image_url   TEXT,
    embedding   VECTOR(768),                                           -- gemini-embedding-001 @ 768-dim, L2-normalized
    status      VARCHAR(32)   DEFAULT 'PENDING',                       -- PENDING | PUBLISHING | PUBLISHED | PARTIAL_FAILURE | REJECTED
    linkedin_status VARCHAR(32) DEFAULT 'PENDING',                     -- per-platform: PENDING | PUBLISHING | PUBLISHED | FAILED
    x_status         VARCHAR(32) DEFAULT 'PENDING',
    linkedin_post_id TEXT,                                             -- LinkedIn post URN from x-restli-id response header
    x_post_id        TEXT,                                             -- X tweet id from data.id
    created_at  TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at  TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP     -- set on every status transition (claim/approve/reject)
);

-- Idempotent migrations for existing installs (CREATE TABLE IF NOT EXISTS
-- does not add new columns to an already-created table).
ALTER TABLE posts ADD COLUMN IF NOT EXISTS linkedin_status VARCHAR(32) DEFAULT 'PENDING';
ALTER TABLE posts ADD COLUMN IF NOT EXISTS x_status         VARCHAR(32) DEFAULT 'PENDING';
ALTER TABLE posts ADD COLUMN IF NOT EXISTS linkedin_post_id TEXT;
ALTER TABLE posts ADD COLUMN IF NOT EXISTS x_post_id        TEXT;
ALTER TABLE posts ADD COLUMN IF NOT EXISTS updated_at       TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP;

-- ── Migration stage: repair legacy rows before the CHECKs are added ───────
-- 1. Backfill per-platform statuses + updated_at for legacy rows so existing
--    installs can be retried/claimed correctly (only ever mirrors the
--    top-level flag for rows that never had per-platform tracking).
UPDATE posts
SET    linkedin_status = CASE WHEN status = 'PUBLISHED' THEN 'PUBLISHED' ELSE 'PENDING' END,
       x_status        = CASE WHEN status = 'PUBLISHED' THEN 'PUBLISHED' ELSE 'PENDING' END,
       updated_at      = COALESCE(created_at, NOW())
WHERE  linkedin_status IS NULL OR x_status IS NULL OR updated_at IS NULL;

-- 2. Mirror the top-level flag onto pre-per-platform-era rows: these carry
--    per-platform rows at the just-added 'PENDING' defaults because they
--    were written before per-platform statuses existed. A durably FAILED or
--    PUBLISHED per-platform status is never overwritten.
UPDATE posts
SET    linkedin_status = 'PUBLISHED',
       x_status        = 'PUBLISHED'
WHERE  status = 'PUBLISHED'
AND    linkedin_status = 'PENDING' AND x_status = 'PENDING';

-- 3. Normalize any out-of-enum statuses / platform values the app can never
--    produce or interpret back to the safe defaults (data-preserving).
UPDATE posts SET status = 'PENDING'
WHERE  status IS NULL OR status NOT IN ('PENDING', 'PUBLISHING', 'PUBLISHED', 'PARTIAL_FAILURE', 'REJECTED');
UPDATE posts SET linkedin_status = 'PENDING'
WHERE  linkedin_status IS NULL OR linkedin_status NOT IN ('PENDING', 'PUBLISHING', 'PUBLISHED', 'FAILED');
UPDATE posts SET x_status = 'PENDING'
WHERE  x_status IS NULL OR x_status NOT IN ('PENDING', 'PUBLISHING', 'PUBLISHED', 'FAILED');
UPDATE posts SET platform = 'both'
WHERE  platform IS NULL OR platform NOT IN ('linkedin', 'x', 'both');

-- 4. Repair top-level completion coherence on legacy rows so the new
--    completion CHECK below is satisfiable. This mirrors the app's own
--    deriveFinalStatus() normalization: all platforms PUBLISHED → PUBLISHED,
--    some → PARTIAL_FAILURE, none → PENDING. Only rows written before the
--    per-platform statuses existed can land incoherently (e.g. top-level
--    PUBLISHED while one platform durably FAILED).
UPDATE posts SET status = 'PENDING'
WHERE  platform = 'both' AND status IN ('PUBLISHED', 'PARTIAL_FAILURE')
   AND linkedin_status <> 'PUBLISHED' AND x_status <> 'PUBLISHED';
UPDATE posts SET status = 'PARTIAL_FAILURE'
WHERE  platform = 'both' AND status = 'PUBLISHED'
   AND (linkedin_status = 'PUBLISHED') <> (x_status = 'PUBLISHED');
UPDATE posts SET status = 'PUBLISHED'
WHERE  platform = 'both' AND status = 'PARTIAL_FAILURE'
   AND linkedin_status = 'PUBLISHED' AND x_status = 'PUBLISHED';
UPDATE posts SET status = 'PENDING'
WHERE  platform <> 'both' AND status IN ('PUBLISHED', 'PARTIAL_FAILURE')
  AND (platform = 'linkedin' AND linkedin_status <> 'PUBLISHED'
       OR platform = 'x' AND x_status <> 'PUBLISHED');
UPDATE posts SET status = 'PUBLISHED'
WHERE  platform <> 'both' AND status = 'PARTIAL_FAILURE'
  AND (platform = 'linkedin' AND linkedin_status = 'PUBLISHED'
       OR platform = 'x' AND x_status = 'PUBLISHED');

-- 5. Pin the embedding column to a fixed 768-dim vector type. The HNSW index
--    below requires known dimensions, and the app always writes 768-dim
--    vectors; any legacy rows with another dimension are unusable for
--    semantic search and are cleared (their other columns stay untouched).
UPDATE posts SET embedding = NULL
WHERE  embedding IS NOT NULL AND vector_dims(embedding) <> 768;
ALTER TABLE posts ALTER COLUMN embedding TYPE vector(768) USING embedding::vector(768);

-- ── Migration stage: NOT NULL timestamps (all rows non-null after step 1) ─
DO $$
BEGIN
    IF EXISTS (SELECT 1 FROM information_schema.columns
               WHERE table_schema = current_schema() AND table_name = 'posts'
                 AND column_name = 'created_at' AND is_nullable = 'YES') THEN
        ALTER TABLE posts ALTER COLUMN created_at SET NOT NULL;
    END IF;
    IF EXISTS (SELECT 1 FROM information_schema.columns
               WHERE table_schema = current_schema() AND table_name = 'posts'
                 AND column_name = 'updated_at' AND is_nullable = 'YES') THEN
        ALTER TABLE posts ALTER COLUMN updated_at SET NOT NULL;
    END IF;
END $$;

-- Strengthen posts_completion_check in place: installs created under this
-- name must get the platform-aware version below, so the old constraint is
-- dropped first (paired drop+re-add stays idempotent).
ALTER TABLE posts DROP CONSTRAINT IF EXISTS posts_completion_check;

-- ── Migration stage: posts integrity constraints (guarded, idempotent) ────
DO $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM pg_constraint WHERE conname = 'posts_platform_check') THEN
        ALTER TABLE posts ADD CONSTRAINT posts_platform_check
            CHECK (platform IN ('linkedin', 'x', 'both'));
    END IF;
    IF NOT EXISTS (SELECT 1 FROM pg_constraint WHERE conname = 'posts_status_check') THEN
        ALTER TABLE posts ADD CONSTRAINT posts_status_check
            CHECK (status IN ('PENDING', 'PUBLISHING', 'PUBLISHED', 'PARTIAL_FAILURE', 'REJECTED'));
    END IF;
    IF NOT EXISTS (SELECT 1 FROM pg_constraint WHERE conname = 'posts_linkedin_status_check') THEN
        ALTER TABLE posts ADD CONSTRAINT posts_linkedin_status_check
            CHECK (linkedin_status IN ('PENDING', 'PUBLISHING', 'PUBLISHED', 'FAILED'));
    END IF;
    IF NOT EXISTS (SELECT 1 FROM pg_constraint WHERE conname = 'posts_x_status_check') THEN
        ALTER TABLE posts ADD CONSTRAINT posts_x_status_check
            CHECK (x_status IN ('PENDING', 'PUBLISHING', 'PUBLISHED', 'FAILED'));
    END IF;
    -- Completion coherence, platform-aware: a PUBLISHED row must have every
    -- platform it targets marked PUBLISHED; a PARTIAL_FAILURE row must have
    -- exactly one successful platform when targeting both, or the single
    -- targeted platform when targeting one. The status of an UNTARGETED
    -- platform never influences the outcome – a stale per-platform flag on
    -- the unused side cannot make a single-platform post appear published or
    -- partially failed. (PUBLISHING/PENDING/REJECTED rows deliberately stay
    -- loose – the webhook legitimately holds top-level PUBLISHING while
    -- per-platform columns are mid-write, and PENDING rows may legitimately
    -- carry FAILED per-platform states after a total-failure round.)
    IF NOT EXISTS (SELECT 1 FROM pg_constraint WHERE conname = 'posts_completion_check') THEN
        ALTER TABLE posts ADD CONSTRAINT posts_completion_check CHECK (
            (status <> 'PUBLISHED' OR (
                (platform = 'x' OR linkedin_status = 'PUBLISHED')
            AND (platform = 'linkedin' OR x_status = 'PUBLISHED')
            ))
        AND (status <> 'PARTIAL_FAILURE' OR (
            CASE platform
                WHEN 'both'     THEN (linkedin_status = 'PUBLISHED') <> (x_status = 'PUBLISHED')
                WHEN 'linkedin' THEN linkedin_status = 'PUBLISHED'
                WHEN 'x'        THEN x_status = 'PUBLISHED'
            END
        ))
        );
    END IF;
END $$;

-- ── Indexes ───────────────────────────────────────────────────────────────
-- HNSW index for fast approximate nearest-neighbor cosine search
CREATE INDEX IF NOT EXISTS posts_embedding_idx
    ON posts USING hnsw (embedding vector_cosine_ops);

-- Cleanup + claim queries are always status-filtered, very often combined
-- with a timestamp range, so one composite per cleanup pattern is cheaper
-- and more useful than bare single-column indexes.
DROP INDEX IF EXISTS posts_status_idx;
DROP INDEX IF EXISTS posts_created_at_idx;
DROP INDEX IF EXISTS posts_updated_at_idx;

CREATE INDEX IF NOT EXISTS posts_status_created_at_idx
    ON posts (status, created_at DESC);     -- cleanup: stale PENDING / PARTIAL_FAILURE scan
CREATE INDEX IF NOT EXISTS posts_status_updated_at_idx
    ON posts (status, updated_at);          -- cleanup: dead PUBLISHING claim recovery scan

-- ============================================================
-- TABLE 1b: research_questions (Episodic Memory – Research Agent)
-- Stores the framing research question derived from a selected
-- topic. topic + question + created_at provide full traceability:
-- every question is linked back to the topic it was framed for
-- and the moment it was generated.
--
-- Lifecycle (status, enforced by DB CHECK): proposed → researching →
-- answered | dropped, with answered → researching allowed for the
-- re-research fallback. "answered_at" records when a session became
-- answered: it is stamped exactly on the answered transition and cleared
-- whenever the session leaves answered, so the two-way DB invariant
-- (answered ⇔ answered_at set) holds for every row. Duplicate
-- topics/questions are intentionally NOT unique: follow-up research on a
-- related-but-different question is always allowed by design.
--
-- The optional `embedding` column is populated for NEW rows so the
-- research-session memory can recognise previously researched topics
-- and questions (pgvector cosine). Legacy rows without an embedding
-- still match exactly, never semantically. When present it must always
-- be 768-dim (CHECK below) so the HNSW index and search queries stay
-- consistent with the app's embedder.
-- ============================================================
CREATE TABLE IF NOT EXISTS research_questions (
    id          UUID          PRIMARY KEY DEFAULT gen_random_uuid(),
    topic       TEXT          NOT NULL,                     -- selected topic title
    question    TEXT          NOT NULL,                     -- primary research question
    aspects     JSONB         NOT NULL DEFAULT '[]',        -- supporting sub-questions
    status      VARCHAR(32)   DEFAULT 'proposed',           -- proposed | researching | answered | dropped
    priority    VARCHAR(16)   DEFAULT 'normal',             -- high | normal | low
    embedding   VECTOR(768)   DEFAULT NULL,                 -- gemini-embedding-001 @ 768-dim (topic + question)
    answered_at TIMESTAMP WITH TIME ZONE DEFAULT NULL,      -- set when a session reaches 'answered'
    created_at  TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Idempotent migration for existing installs: add the answered_at column.
ALTER TABLE research_questions ADD COLUMN IF NOT EXISTS answered_at TIMESTAMP WITH TIME ZONE;

-- ── Migration stage: repair legacy rows before the CHECKs are added ───────
-- Normalize out-of-enum values the app can never produce (data-preserving).
UPDATE research_questions SET status = 'proposed'
WHERE  status IS NULL OR status NOT IN ('proposed', 'researching', 'answered', 'dropped');
UPDATE research_questions SET priority = 'normal'
WHERE  priority IS NULL OR priority NOT IN ('high', 'normal', 'low');
UPDATE research_questions SET aspects = '[]' WHERE aspects IS NULL;

-- "answered" sessions must carry a timestamp; legacy rows never recorded
-- one, so backfill from created_at (keeps the row meaningful, adds info).
-- Conversely, a non-answered session must NOT carry one: leaving "answered"
-- (re-research / dropped) clears it, and no legacy row ever set it outside
-- "answered".
UPDATE research_questions SET answered_at = created_at
WHERE  status = 'answered' AND answered_at IS NULL;
UPDATE research_questions SET answered_at = NULL
WHERE  status <> 'answered' AND answered_at IS NOT NULL;

-- Embeddings of the wrong dimension cannot be searched by the HNSW index or
-- the semantic-memory queries; clear them so the dims CHECK can be added
-- (the row, topic/question/aspects/status stay untouched). The column is
-- then pinned to a fixed 768-dim type like its app-written counterpart.
UPDATE research_questions SET embedding = NULL
WHERE  embedding IS NOT NULL AND vector_dims(embedding) <> 768;
ALTER TABLE research_questions ALTER COLUMN embedding TYPE vector(768) USING embedding::vector(768);

-- ── Migration stage: NOT NULL aspects ─────────────────────────────────────
DO $$
BEGIN
    IF EXISTS (SELECT 1 FROM information_schema.columns
               WHERE table_schema = current_schema() AND table_name = 'research_questions'
                 AND column_name = 'aspects' AND is_nullable = 'YES') THEN
        ALTER TABLE research_questions ALTER COLUMN aspects SET NOT NULL;
    END IF;
END $$;

-- ── Migration stage: research_questions integrity constraints ─────────────
DO $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM pg_constraint WHERE conname = 'research_questions_status_check') THEN
        ALTER TABLE research_questions ADD CONSTRAINT research_questions_status_check
            CHECK (status IN ('proposed', 'researching', 'answered', 'dropped'));
    END IF;
    IF NOT EXISTS (SELECT 1 FROM pg_constraint WHERE conname = 'research_questions_priority_check') THEN
        ALTER TABLE research_questions ADD CONSTRAINT research_questions_priority_check
            CHECK (priority IN ('high', 'normal', 'low'));
    END IF;
    -- Two-way lifecycle invariant: a session is answered exactly when
    -- answered_at is set. The one-way version installs may still carry under
    -- this name, so it is dropped first and re-created strengthened
    -- (the DROP + guarded ADD pair is idempotent; on a fresh DB the DROP is
    -- a no-op).
    ALTER TABLE research_questions DROP CONSTRAINT IF EXISTS research_questions_answered_at_check;
    IF NOT EXISTS (SELECT 1 FROM pg_constraint WHERE conname = 'research_questions_answered_at_check') THEN
        ALTER TABLE research_questions ADD CONSTRAINT research_questions_answered_at_check
            CHECK ((status = 'answered') = (answered_at IS NOT NULL));
    END IF;
    IF NOT EXISTS (SELECT 1 FROM pg_constraint WHERE conname = 'research_questions_embedding_dims_check') THEN
        ALTER TABLE research_questions ADD CONSTRAINT research_questions_embedding_dims_check
            CHECK (embedding IS NULL OR vector_dims(embedding) = 768);
    END IF;
END $$;

CREATE INDEX IF NOT EXISTS research_questions_created_at_idx
    ON research_questions (created_at DESC);

CREATE INDEX IF NOT EXISTS research_questions_embedding_idx
    ON research_questions USING hnsw (embedding vector_cosine_ops);

-- ============================================================
-- TABLE 1c: research_sources (Episodic Memory – Research Agent)
-- One row per normalized, deduplicated source gathered for a
-- research session, linked back to its research_questions row.
--
-- The stored `url` is the raw provider URL; `canonical_url` is a
-- deterministic fingerprint produced by the same normalizer the app uses
-- for its dedup keys (lowercase scheme/host, www. stripped, fragment and
-- utm_* tracking params dropped, trailing slash normalized – see
-- src.ingestion._normalize_url). The app writes canonical_url on insert;
-- a partial UNIQUE index then DB-enforces what the app otherwise defends in
-- Python: one full-content source per (research session, canonical URL).
-- Link rows (empty body – see below) are exempt: they are contentless
-- per-session references and may coexist with the full record.
--
-- Access-history dedup across sessions stays at the application layer
-- (src.memory) and is unchanged: re-searching a URL in a DIFFERENT session
-- is a fresh, legitimate source row; worst case the old row is re-linked.
--
-- Each row belongs to exactly one session (no many-to-many link table).
-- When a run discovers sources that are ALL already known from earlier
-- sessions, src.memory.link_research_sources writes lightweight per-session
-- *link rows* (url + canonical_url + title only, empty body) instead of
-- duplicating the full source content; hydration then re-loads the full
-- content from the newest stored record with the same normalized url.
-- ============================================================
CREATE TABLE IF NOT EXISTS research_sources (
    id                   UUID          PRIMARY KEY DEFAULT gen_random_uuid(),
    research_question_id UUID          REFERENCES research_questions(id) ON DELETE CASCADE,
    url                  TEXT          NOT NULL,
    canonical_url        TEXT,                                      -- normalized fingerprint (see table comment)
    title                TEXT          DEFAULT '',
    body                 TEXT          DEFAULT '',
    source               TEXT          DEFAULT '',
    published            TEXT          DEFAULT '',
    score                DOUBLE PRECISION DEFAULT 0,
    source_type          VARCHAR(32)   DEFAULT 'secondary',         -- primary | secondary
    accessed_at          TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- ============================================================
-- FUNCTION: canonicalize_source_url(raw)
-- Best-effort SQL mirror of src.ingestion._normalize_url used ONLY to
-- backfill canonical_url for rows written before this column existed.
-- New writes always carry the app-side fingerprint, so backfilled values
-- only ever need to be internally consistent for dedup, not byte-identical
-- to the Python normalizer.
-- ============================================================
CREATE OR REPLACE FUNCTION canonicalize_source_url(raw TEXT) RETURNS TEXT AS $$
DECLARE
    u               TEXT;
    scheme          TEXT;
    rest            TEXT;
    host            TEXT;
    path_and_query  TEXT;
    path            TEXT;
    query           TEXT;
BEGIN
    IF raw IS NULL THEN
        RETURN NULL;
    END IF;
    u := lower(btrim(raw));
    IF u = '' THEN
        RETURN '';
    END IF;
    -- scheme (default https, matching _normalize_url)
    IF u LIKE '%://%' THEN
        scheme := split_part(u, '://', 1);
        rest   := substr(u, strpos(u, '://') + 3);
    ELSE
        scheme := 'https';
        rest   := u;
    END IF;
    -- fragment is always dropped
    IF POSITION('#' IN rest) > 0 THEN
        rest := substr(rest, 1, POSITION('#' IN rest) - 1);
    END IF;
    -- host vs path+query
    IF POSITION('/' IN rest) > 0 THEN
        host            := substr(rest, 1, POSITION('/' IN rest) - 1);
        path_and_query  := substr(rest, POSITION('/' IN rest));
    ELSE
        host            := rest;
        path_and_query  := '/';
    END IF;
    IF host LIKE 'www.%' THEN
        host := substr(host, 5);
    END IF;
    -- split path?query and drop utm_* tracking params (case-insensitive name).
    -- The '?' is kept in `query` so the separator survives the param removal.
    IF POSITION('?' IN path_and_query) > 0 THEN
        path  := substr(path_and_query, 1, POSITION('?' IN path_and_query) - 1);
        query := substr(path_and_query, POSITION('?' IN path_and_query));
        query := regexp_replace(query, '([?&])utm_[^&=]*=[^&]*', '\1', 'g');
        query := regexp_replace(query, '^[?&]+', '?', 'g');
        query := regexp_replace(query, '&{2,}', '&', 'g');
        query := regexp_replace(query, '[?&]$', '', 'g');
    ELSE
        path  := path_and_query;
        query := '';
    END IF;
    -- trailing slash is normalized (except for the bare root)
    path := rtrim(path, '/');
    IF path = '' THEN
        path := '/';
    END IF;
    u := scheme || '://' || host || path;
    IF query <> '' THEN
        u := u || query;            -- query already carries its leading '?'
    END IF;
    RETURN u;
END;
$$ LANGUAGE plpgsql IMMUTABLE;

-- Migration stage: add + backfill canonical_url for existing installs.
ALTER TABLE research_sources ADD COLUMN IF NOT EXISTS canonical_url TEXT;
UPDATE research_sources SET canonical_url = canonicalize_source_url(url)
WHERE  canonical_url IS NULL;

-- Merge legacy duplicate sources per (session, canonical URL). Without this
-- step the unique index below could not be created on existing data.
--
-- What is deleted and why: only ROWS INSIDE research_sources that repeat the
-- same canonical_url within the same research_question_id session. Exactly
-- one row survives per (research_question_id, canonical_url) fingerprint –
-- the RICHEST one (prefers a non-empty body, then the newest access). No
-- research session ever loses its canonical source record: every fingerprint
-- that existed before still has a surviving row afterwards.
--
-- No references are orphaned or "migrated": research_sources is a LEAF table
-- – nothing references it (its own FK to research_questions is the only link,
-- and fires ON DELETE CASCADE from the QUESTION side). Deleting a duplicate
-- source row therefore cannot cascade to any other record, and the surviving
-- row keeps the session's reference to that canonical source. Removed rows
-- are true duplicates (same session + same canonical_url), not distinct
-- sources: any content they carried was one of several copies of the same
-- record, and the richest retained copy is what hydration loads.
DELETE FROM research_sources
WHERE id IN (
    SELECT id FROM (
        SELECT id,
               ROW_NUMBER() OVER (
                   PARTITION BY research_question_id, canonical_url
                   ORDER BY (body IS NOT NULL AND body <> '') DESC,
                            accessed_at DESC NULLS LAST,
                            id
               ) AS rn
        FROM research_sources
        WHERE canonical_url <> ''
    ) ranked
    WHERE rn > 1
);

-- canonical_url is required for every (new and legacy) row.
DO $$
BEGIN
    IF EXISTS (SELECT 1 FROM information_schema.columns
               WHERE table_schema = current_schema() AND table_name = 'research_sources'
                 AND column_name = 'canonical_url' AND is_nullable = 'YES') THEN
        ALTER TABLE research_sources ALTER COLUMN canonical_url SET NOT NULL;
    END IF;
END $$;

-- Normalize any out-of-enum source_type before its CHECK is added.
UPDATE research_sources SET source_type = 'secondary'
WHERE  source_type IS NULL OR source_type NOT IN ('primary', 'secondary');

-- Uniqueness: at most ONE full-content source per (research_question_id,
-- canonical_url). link rows (empty body) carry no content and are exempt.
CREATE UNIQUE INDEX IF NOT EXISTS research_sources_session_url_uniq
    ON research_sources (research_question_id, canonical_url)
    WHERE body IS DISTINCT FROM '';

-- Migration stage: research_sources integrity constraints (guarded).
DO $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM pg_constraint WHERE conname = 'research_sources_source_type_check') THEN
        ALTER TABLE research_sources ADD CONSTRAINT research_sources_source_type_check
            CHECK (source_type IN ('primary', 'secondary'));
    END IF;
END $$;

CREATE INDEX IF NOT EXISTS research_sources_question_idx
    ON research_sources (research_question_id);

CREATE INDEX IF NOT EXISTS research_sources_url_idx
    ON research_sources (url);

-- ============================================================
-- TABLE 2: style_profile (Semantic Memory)
-- Stores a single JSONB document representing the master
-- writing style profile derived from historical posts.
-- CONSTRAINT ensures only one row ever exists (id = 1).
-- ============================================================
CREATE TABLE IF NOT EXISTS style_profile (
    id         INT           PRIMARY KEY DEFAULT 1,
    profile    JSONB         NOT NULL,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT single_row CHECK (id = 1)
);

-- ============================================================
-- TABLE 3: system_config (Auth Token Tracking)
-- Key-value store for system-level configuration such as
-- LinkedIn token expiry, webhook secrets, etc.
-- ============================================================
CREATE TABLE IF NOT EXISTS system_config (
    key        VARCHAR(64)   PRIMARY KEY,
    value      TEXT          NOT NULL,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- ============================================================
-- FUNCTION: cleanup_stale_drafts()
-- Resolves drafts that never reached a terminal state:
--   - PENDING drafts older than 24 hours (never approved)          → REJECTED
--   - PARTIAL_FAILURE drafts older than 24 hours (approval was
--     given but a platform outage was never resolved)              → REJECTED
--   - PUBLISHING rows whose claim never completed (webhook crash /
--     timeout). updated_at is set by the webhook at claim time, so a row
--     still PUBLISHING after 15 minutes is safely a dead claim (the publish
--     flow touches the DB within seconds). It is reverted to PENDING so the
--     next ✅ Approve tap reclaims it and publishes ONLY the missing
--     platforms (idempotent – nothing already PUBLISHED is re-published).
-- ============================================================
CREATE OR REPLACE FUNCTION cleanup_stale_drafts()
RETURNS void AS $$
BEGIN
    UPDATE posts
    SET    status = 'REJECTED', updated_at = NOW()
    WHERE  status IN ('PENDING', 'PARTIAL_FAILURE')
    AND    created_at < NOW() - INTERVAL '24 hours';

    -- Recover claims that never resolved (dead webhook instance).
    UPDATE posts
    SET    status = 'PENDING', updated_at = NOW()
    WHERE  status = 'PUBLISHING'
    AND    updated_at < NOW() - INTERVAL '15 minutes';

    RAISE NOTICE 'Stale draft cleanup completed at %', NOW();
END;
$$ LANGUAGE plpgsql;

-- ============================================================
-- Cleanup scheduling
-- ============================================================
-- cleanup_stale_drafts() is scheduled OUTSIDE this file by the project's
-- declarative automation: .github/workflows/keepalive.yml invokes it daily
-- at 08:00 UTC via the Supabase RPC endpoint (sb.rpc("cleanup_stale_drafts"))
-- alongside the LinkedIn token canary. It can also be run manually from the
-- SQL editor (SELECT cleanup_stale_drafts();).
--
-- (An internal pg_cron schedule is deliberately NOT configured here: the
-- repository deploys to Supabase where the pg_cron extension is not enabled,
-- and an unguarded cron.schedule() call would fail the whole script on any
-- fresh deploy. If you self-host with pg_cron available, uncomment and adjust
-- the schedule below instead of the GitHub Action.)
--
-- CREATE EXTENSION IF NOT EXISTS pg_cron;
-- SELECT cron.schedule(
--     'cleanup-stale-drafts',
--     '0 * * * *',                  -- hourly at the top of the hour
--     'SELECT cleanup_stale_drafts();'
-- );

-- ============================================================
-- Seed: Default style profile (update via agent after onboarding)
-- ============================================================
INSERT INTO style_profile (id, profile)
VALUES (
    1,
    '{
        "tone": "professional yet conversational",
        "voice": "first-person, thought-leader",
        "linkedin": {
            "avg_length": 250,
            "use_emojis": true,
            "use_hashtags": true,
            "hashtag_count": 3,
            "cta_style": "question-based",
            "structure": "hook → insight → personal take → CTA"
        },
        "x": {
            "avg_length": 220,
            "use_emojis": true,
            "use_hashtags": true,
            "hashtag_count": 2,
            "style": "punchy, direct, thread-friendly"
        },
        "topics_of_interest": ["AI", "LLMs", "developer tools", "AWS", "Cloud Computing", "startups", "Python", "Automation"],
        "avoid": ["clickbait", "excessive exclamation marks", "unsubstantiated claims"]
    }'
)
ON CONFLICT (id) DO NOTHING;

-- ============================================================
-- Seed: Default system config entries
-- ============================================================
INSERT INTO system_config (key, value) VALUES
    ('schema_version',          '2.0.0'),
    ('linkedin_token_status',   'UNKNOWN'),
    ('last_pipeline_run',       'NEVER')
ON CONFLICT (key) DO NOTHING;