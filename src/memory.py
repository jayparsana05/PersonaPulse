"""
PersonaPulse – Memory Module
============================
Handles vector embeddings (Gemini gemini-embedding-001) with
L2 normalization, and semantic deduplication against the
Supabase posts table via pgvector cosine distance queries.

Key Functions
-------------
- get_normalized_embedding(text)  → list[float]
- check_is_duplicate(embedding, threshold)  → bool
- store_draft(platform, topic, content, embedding, article_url, image_url) → str  (UUID)
- update_post_status(post_id, status)
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
from google import genai
from google.genai import types as genai_types
from supabase import create_client, Client

from src.config import settings

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Clients (initialized lazily at module level for reuse)
# ---------------------------------------------------------------------------

_gemini_client: genai.Client | None = None
_supabase_client: Client | None = None


def _get_gemini() -> genai.Client:
    global _gemini_client
    if _gemini_client is None:
        _gemini_client = genai.Client(api_key=settings.GEMINI_API_KEY)
    return _gemini_client


def _get_supabase() -> Client:
    global _supabase_client
    if _supabase_client is None:
        _supabase_client = create_client(
            settings.SUPABASE_URL,
            settings.SUPABASE_SERVICE_ROLE_KEY,
        )
    return _supabase_client


# ---------------------------------------------------------------------------
# Public: Embedding with L2 Normalization
# ---------------------------------------------------------------------------

def get_normalized_embedding(text: str) -> list[float]:
    """
    Generate a 768-dimensional, L2-normalized embedding for *text*
    using Google's gemini-embedding-001 model.

    Why normalize?
    --------------
    The Gemini API truncates the raw output vector to 768 dims,
    which breaks the original unit-length guarantee.  Dividing by
    the L2 norm (||v|| = sqrt(Σ xᵢ²)) restores magnitude 1.0, which
    is a precondition for meaningful cosine-similarity comparisons.
    """
    client = _get_gemini()

    response = client.models.embed_content(
        model=settings.EMBEDDING_MODEL,
        contents=text,
        config=genai_types.EmbedContentConfig(
            output_dimensionality=settings.EMBEDDING_DIMENSIONS,
        ),
    )

    raw_vector: list[float] = response.embeddings[0].values

    # ── L2 Normalization ────────────────────────────────────────────────
    vec = np.array(raw_vector, dtype=np.float64)
    norm = np.linalg.norm(vec)                   # sqrt(Σ xᵢ²)

    if norm < 1e-10:
        log.warning("[Memory] Embedding norm is near-zero – returning zero vector.")
        return raw_vector                        # edge-case: empty / garbage text

    normalized = (vec / norm).tolist()
    log.debug("[Memory] Embedding norm before=%.6f  after=%.6f", norm, np.linalg.norm(normalized))
    return normalized


# ---------------------------------------------------------------------------
# Public: Semantic Deduplication
# ---------------------------------------------------------------------------

def check_is_duplicate(
    embedding: list[float],
    threshold: float = 0.85,
) -> bool:
    """
    Return True if a post with cosine similarity > *threshold* already
    exists in the `posts` table (covers PUBLISHED and PENDING statuses).

    Cosine *distance* in pgvector: 0 = identical, 2 = opposite.
    Cosine *similarity* = 1 - cosine_distance.
    We filter where  (1 - distance) > threshold  ↔  distance < (1 - threshold).
    """
    supabase = _get_supabase()
    distance_cutoff = 1.0 - threshold            # e.g. 0.15 for threshold=0.85

    # Format the embedding as a PostgreSQL vector literal
    vector_literal = "[" + ",".join(f"{x:.8f}" for x in embedding) + "]"

    # Use raw SQL via RPC to leverage the pgvector operator
    result = supabase.rpc(
        "match_posts_by_embedding",
        {
            "query_embedding": vector_literal,
            "match_threshold": distance_cutoff,
            "match_count": 1,
        },
    ).execute()

    if result.data:
        log.info(
            "[Memory] Duplicate detected – similar post id=%s",
            result.data[0].get("id", "unknown"),
        )
        return True

    return False


# ---------------------------------------------------------------------------
# Public: Store Draft Post
# ---------------------------------------------------------------------------

def store_draft(
    platform: str,
    topic: str,
    content: str,
    embedding: list[float],
    article_url: Optional[str] = None,
    image_url: Optional[str] = None,
) -> str:
    """
    Insert a new PENDING draft into the `posts` table.
    Returns the generated UUID of the new row.
    """
    supabase = _get_supabase()

    vector_literal = "[" + ",".join(f"{x:.8f}" for x in embedding) + "]"

    row = {
        "platform": platform,
        "topic": topic,
        "content": content,
        "article_url": article_url,
        "image_url": image_url,
        "embedding": vector_literal,
        "status": "PENDING",
    }

    result = supabase.table("posts").insert(row).execute()
    post_id: str = result.data[0]["id"]
    log.info("[Memory] Draft stored – id=%s  platform=%s", post_id, platform)
    return post_id


# ---------------------------------------------------------------------------
# Public: Update Post Status
# ---------------------------------------------------------------------------

def update_post_status(post_id: str, status: str) -> None:
    """Update the status column for a given post UUID."""
    supabase = _get_supabase()
    supabase.table("posts").update({"status": status}).eq("id", post_id).execute()
    log.info("[Memory] Post %s → status=%s", post_id, status)


# ---------------------------------------------------------------------------
# Public: Retrieve Style Profile
# ---------------------------------------------------------------------------

def get_style_profile() -> dict:
    """
    Fetch the master style profile JSON from the `style_profile` table.
    Returns an empty dict if no profile exists yet.
    """
    supabase = _get_supabase()
    result = supabase.table("style_profile").select("profile").eq("id", 1).execute()

    if result.data:
        return result.data[0]["profile"]

    log.warning("[Memory] No style_profile found – using empty defaults.")
    return {}


# ---------------------------------------------------------------------------
# Supabase RPC helper SQL (run once during setup)
# ---------------------------------------------------------------------------
# The check_is_duplicate function calls an RPC named match_posts_by_embedding.
# Add this SQL function to your Supabase project via the SQL editor:
#
#   CREATE OR REPLACE FUNCTION match_posts_by_embedding(
#       query_embedding   VECTOR(768),
#       match_threshold   FLOAT,
#       match_count       INT
#   )
#   RETURNS TABLE (id UUID, cosine_distance FLOAT) AS $$
#   BEGIN
#       RETURN QUERY
#       SELECT
#           p.id,
#           (p.embedding <=> query_embedding)::FLOAT AS cosine_distance
#       FROM posts p
#       WHERE p.status IN ('PUBLISHED', 'PENDING')
#         AND (p.embedding <=> query_embedding) < match_threshold
#       ORDER BY cosine_distance ASC
#       LIMIT match_count;
#   END;
#   $$ LANGUAGE plpgsql;
