-- ============================================================
-- PersonaPulse AI Social Media Agent - Database Schema
-- ============================================================
-- Run this against your Supabase PostgreSQL instance.
-- Requires the pgvector extension to be enabled.

-- Enable pgvector extension for semantic similarity search
CREATE EXTENSION IF NOT EXISTS vector;

-- Enable pg_cron extension for scheduled cleanup (if available)
-- CREATE EXTENSION IF NOT EXISTS pg_cron;

-- ============================================================
-- TABLE 1: posts (Episodic Memory)
-- Stores all published posts, pending drafts, and historical
-- examples used for style learning and deduplication.
-- ============================================================
CREATE TABLE IF NOT EXISTS posts (
    id          UUID          PRIMARY KEY DEFAULT gen_random_uuid(),
    platform    VARCHAR(32)   NOT NULL,                                -- 'linkedin', 'x', or 'both'
    topic       VARCHAR(256)  NOT NULL,
    content     TEXT          NOT NULL,
    article_url TEXT,
    image_url   TEXT,
    embedding   VECTOR(768),                                           -- gemini-embedding-001 @ 768-dim, L2-normalized
    status      VARCHAR(32)   DEFAULT 'PENDING',                       -- PENDING | PUBLISHED | REJECTED | PARTIAL_FAILURE
    created_at  TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- HNSW index for fast approximate nearest-neighbor cosine search
CREATE INDEX IF NOT EXISTS posts_embedding_idx
    ON posts USING hnsw (embedding vector_cosine_ops);

-- Index for efficient status-based filtering (used by cleanup routine)
CREATE INDEX IF NOT EXISTS posts_status_idx ON posts (status);
CREATE INDEX IF NOT EXISTS posts_created_at_idx ON posts (created_at DESC);

-- ============================================================
-- TABLE 1b: research_questions (Episodic Memory – Research Agent)
-- Stores the framing research question derived from a selected
-- topic. topic + question + created_at provide full traceability:
-- every question is linked back to the topic it was framed for
-- and the moment it was generated.
--
-- The optional `embedding` column is populated for NEW rows so the
-- research-session memory can recognise previously researched topics
-- and questions (pgvector cosine). Legacy rows without an embedding
-- still match exactly, never semantically.
-- ============================================================
CREATE TABLE IF NOT EXISTS research_questions (
    id          UUID          PRIMARY KEY DEFAULT gen_random_uuid(),
    topic       TEXT          NOT NULL,                     -- selected topic title
    question    TEXT          NOT NULL,                     -- primary research question
    aspects     JSONB         DEFAULT '[]',                 -- supporting sub-questions
    status      VARCHAR(32)   DEFAULT 'proposed',           -- proposed | researching | answered | dropped
    priority    VARCHAR(16)   DEFAULT 'normal',             -- high | normal | low
    embedding   VECTOR(768)   DEFAULT NULL,                 -- gemini-embedding-001 @ 768-dim (topic + question)
    created_at  TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS research_questions_created_at_idx
    ON research_questions (created_at DESC);

CREATE INDEX IF NOT EXISTS research_questions_embedding_idx
    ON research_questions USING hnsw (embedding vector_cosine_ops);

-- ============================================================
-- TABLE 1c: research_sources (Episodic Memory – Research Agent)
-- One row per normalized, deduplicated source gathered for a
-- research session, linked back to its research_questions row.
-- `url` is stored normalized; callers add a per-session source
-- cap before persistence. Access-history dedup across sessions is
-- done at the application layer (see src.memory).
-- ============================================================
CREATE TABLE IF NOT EXISTS research_sources (
    id                   UUID          PRIMARY KEY DEFAULT gen_random_uuid(),
    research_question_id UUID          REFERENCES research_questions(id) ON DELETE CASCADE,
    url                  TEXT          NOT NULL,
    title                TEXT          DEFAULT '',
    body                 TEXT          DEFAULT '',
    source               TEXT          DEFAULT '',
    published            TEXT          DEFAULT '',
    score                DOUBLE PRECISION DEFAULT 0,
    source_type          VARCHAR(32)   DEFAULT 'secondary',     -- primary | secondary
    accessed_at          TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

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
-- Auto-rejects any PENDING drafts older than 24 hours.
-- Should be scheduled via pg_cron or called by a GitHub Action.
-- ============================================================
CREATE OR REPLACE FUNCTION cleanup_stale_drafts()
RETURNS void AS $$
BEGIN
    UPDATE posts
    SET    status = 'REJECTED'
    WHERE  status = 'PENDING'
    AND    created_at < NOW() - INTERVAL '24 hours';

    RAISE NOTICE 'Stale draft cleanup completed at %', NOW();
END;
$$ LANGUAGE plpgsql;

-- ============================================================
-- pg_cron Schedule (uncomment if pg_cron extension is enabled)
-- Runs cleanup_stale_drafts() every hour.
-- ============================================================
SELECT cron.schedule(
    'cleanup-stale-drafts',     -- job name
    '0 0 * * 0',                -- cron expression: once a week (Sunday at midnight)
    'SELECT cleanup_stale_drafts();'
);

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
    ('schema_version',          '1.0.0'),
    ('linkedin_token_status',   'UNKNOWN'),
    ('last_pipeline_run',       'NEVER')
ON CONFLICT (key) DO NOTHING;
