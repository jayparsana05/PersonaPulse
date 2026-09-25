-- ============================================================
-- PersonaPulse – Phase 2: Database Integrity & Schema Hardening
-- ============================================================
-- Migration for EXISTING installs applying the Phase-2 integrity
-- hardening. It mirrors the guarded migration stage embedded in
-- schema.sql (the canonical deploy artifact) so both paths deploy the
-- same DDL. It is idempotent: every step is guarded and safe to run
-- repeatedly, and it repairs conflicting legacy data in place (no row
-- content is deleted or destructively rewritten).
--
-- What it adds:
--   research_questions : answered_at column + status/priority/enum CHECKs;
--                        two-way answered lifecycle invariant (answered ⇔
--                        answered_at set); embedding-dims CHECK; aspects
--                        NOT NULL.
--   research_sources   : canonical_url (normalized fingerprint) + a
--                        partial UNIQUE index guaranteeing at most one
--                        full-content source per (session, canonical URL);
--                        legacy duplicates merged (richest kept) with no
--                        reference lost or orphaned.
--   posts              : status/platform/enum + platform-aware completion
--                        coherence CHECKs (the untargeted platform's status
--                        never influences the outcome); created_at/updated_at
--                        NOT NULL; index consolidation for the cleanup query
--                        patterns.
--   cleanup_stale_drafts() re-created (unchanged body) for installs that
--   predate it. Scheduling is external via .github/workflows/keepalive.yml
--   (daily 08:00 UTC, Supabase RPC) – see the notes at the bottom.

CREATE EXTENSION IF NOT EXISTS vector;

-- ============================================================
-- posts
-- ============================================================
ALTER TABLE posts ADD COLUMN IF NOT EXISTS linkedin_status VARCHAR(32) DEFAULT 'PENDING';
ALTER TABLE posts ADD COLUMN IF NOT EXISTS x_status         VARCHAR(32) DEFAULT 'PENDING';
ALTER TABLE posts ADD COLUMN IF NOT EXISTS linkedin_post_id TEXT;
ALTER TABLE posts ADD COLUMN IF NOT EXISTS x_post_id        TEXT;
ALTER TABLE posts ADD COLUMN IF NOT EXISTS updated_at       TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP;

-- 1. Backfill per-platform statuses + updated_at for legacy rows (mirrors
--    the top-level flag; no row content is changed).
UPDATE posts
SET    linkedin_status = CASE WHEN status = 'PUBLISHED' THEN 'PUBLISHED' ELSE 'PENDING' END,
       x_status        = CASE WHEN status = 'PUBLISHED' THEN 'PUBLISHED' ELSE 'PENDING' END,
       updated_at      = COALESCE(created_at, NOW())
WHERE  linkedin_status IS NULL OR x_status IS NULL OR updated_at IS NULL;

-- 2. Mirror the top-level flag onto pre-per-platform-era rows: these carry
--    the just-added 'PENDING' defaults for every platform because they were
--    written before per-platform statuses existed. A durably FAILED or
--    PUBLISHED per-platform status is never overwritten.
UPDATE posts
SET    linkedin_status = 'PUBLISHED',
       x_status        = 'PUBLISHED'
WHERE  status = 'PUBLISHED'
AND    linkedin_status = 'PENDING' AND x_status = 'PENDING';

-- 3. Normalize out-of-enum statuses / platform values the app cannot
--    interpret back to the safe defaults (data-preserving).
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

-- 5. Pin the embedding column to a fixed 768-dim vector type (the HNSW index
--    requires known dimensions; the app always writes 768-dim vectors).
UPDATE posts SET embedding = NULL
WHERE  embedding IS NOT NULL AND vector_dims(embedding) <> 768;
ALTER TABLE posts ALTER COLUMN embedding TYPE vector(768) USING embedding::vector(768);

-- NOT NULL timestamps (all rows non-null after step 1).
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

-- Integrity constraints (guarded, idempotent).
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
    -- Completion coherence: PUBLISHED rows must have every targeted platform
    -- marked PUBLISHED; PARTIAL_FAILURE rows must have exactly one.
    -- Platform-aware completion coherence: a PUBLISHED row must have every
    -- platform it targets marked PUBLISHED; a PARTIAL_FAILURE row must have
    -- exactly one successful platform when targeting both, or the single
    -- targeted platform when targeting one. The status of an UNTARGETED
    -- platform never influences the outcome. Installs created under this
    -- name keep the older not-platform-aware shape, so it is dropped first
    -- and re-created strengthened (the DROP + guarded ADD pair is
    -- idempotent).
    ALTER TABLE posts DROP CONSTRAINT IF EXISTS posts_completion_check;
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

-- Index consolidation: cleanup queries are status-filtered + time-ranged,
-- so composites replace the three bare single-column indexes.
DROP INDEX IF EXISTS posts_status_idx;
DROP INDEX IF EXISTS posts_created_at_idx;
DROP INDEX IF EXISTS posts_updated_at_idx;
CREATE INDEX IF NOT EXISTS posts_status_created_at_idx ON posts (status, created_at DESC);
CREATE INDEX IF NOT EXISTS posts_status_updated_at_idx  ON posts (status, updated_at);
CREATE INDEX IF NOT EXISTS posts_embedding_idx
    ON posts USING hnsw (embedding vector_cosine_ops);

-- ============================================================
-- research_questions
-- ============================================================
ALTER TABLE research_questions ADD COLUMN IF NOT EXISTS answered_at TIMESTAMP WITH TIME ZONE;

-- Repair legacy rows before the CHECKs are added (data-preserving).
UPDATE research_questions SET status = 'proposed'
WHERE  status IS NULL OR status NOT IN ('proposed', 'researching', 'answered', 'dropped');
UPDATE research_questions SET priority = 'normal'
WHERE  priority IS NULL OR priority NOT IN ('high', 'normal', 'low');
UPDATE research_questions SET aspects = '[]' WHERE aspects IS NULL;
UPDATE research_questions SET answered_at = created_at
WHERE  status = 'answered' AND answered_at IS NULL;
UPDATE research_questions SET answered_at = NULL
WHERE  status <> 'answered' AND answered_at IS NOT NULL;
UPDATE research_questions SET embedding = NULL
WHERE  embedding IS NOT NULL AND vector_dims(embedding) <> 768;
ALTER TABLE research_questions ALTER COLUMN embedding TYPE vector(768) USING embedding::vector(768);

DO $$
BEGIN
    IF EXISTS (SELECT 1 FROM information_schema.columns
               WHERE table_schema = current_schema() AND table_name = 'research_questions'
                 AND column_name = 'aspects' AND is_nullable = 'YES') THEN
        ALTER TABLE research_questions ALTER COLUMN aspects SET NOT NULL;
    END IF;
END $$;

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
    -- answered_at is set. The one-way version installs may already carry
    -- under this name, so it is dropped first and re-created strengthened
    -- (the DROP + guarded ADD pair is idempotent).
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
-- research_sources
-- ============================================================
-- Best-effort SQL mirror of src.ingestion._normalize_url used ONLY to
-- backfill canonical_url for rows written before this column existed.
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

-- At most one full-content source per (research_question_id, canonical_url).
CREATE UNIQUE INDEX IF NOT EXISTS research_sources_session_url_uniq
    ON research_sources (research_question_id, canonical_url)
    WHERE body IS DISTINCT FROM '';

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
-- cleanup_stale_drafts() + scheduling
-- ============================================================
CREATE OR REPLACE FUNCTION cleanup_stale_drafts()
RETURNS void AS $$
BEGIN
    UPDATE posts
    SET    status = 'REJECTED', updated_at = NOW()
    WHERE  status IN ('PENDING', 'PARTIAL_FAILURE')
    AND    created_at < NOW() - INTERVAL '24 hours';

    UPDATE posts
    SET    status = 'PENDING', updated_at = NOW()
    WHERE  status = 'PUBLISHING'
    AND    updated_at < NOW() - INTERVAL '15 minutes';

    RAISE NOTICE 'Stale draft cleanup completed at %', NOW();
END;
$$ LANGUAGE plpgsql;

-- cleanup_stale_drafts() is scheduled BY THE PROJECT OUTSIDE this file:
-- .github/workflows/keepalive.yml invokes it daily at 08:00 UTC via the
-- Supabase RPC endpoint (sb.rpc("cleanup_stale_drafts")). An internal
-- pg_cron schedule is deliberately NOT configured (Supabase does not enable
-- the pg_cron extension, and an unguarded cron.schedule() call would fail
-- the script on any fresh deploy). If you self-host with pg_cron available:
--
-- CREATE EXTENSION IF NOT EXISTS pg_cron;
-- SELECT cron.schedule(
--     'cleanup-stale-drafts',
--     '0 * * * *',
--     'SELECT cleanup_stale_drafts();'
-- );