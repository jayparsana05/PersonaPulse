"""
Unit tests for src.publishers (LinkedIn / X post-ID extraction).
==============================================================

Regression coverage for the per-platform post id used by the publishing
workflow's deduplication:

- LinkedIn identifies the created post in the ``x-restli-id`` RESPONSE HEADER
  (a URN such as ``urn:li:ugcPost:...``) — the older ``x-linkedin-id`` is NOT
  documented and must not be relied on.
- X returns ``data.id`` in the created-tweet response body.

Network I/O is mocked (httpx / tweepy), so no live API calls are made.
Dummy env vars are installed before importing src.publishers so config
validation passes without a populated .env file.
"""

from __future__ import annotations

import os
import unittest
from types import SimpleNamespace
from unittest.mock import patch

_REQUIRED_ENV = {
    "GEMINI_API_KEY": "test-gemini",
    "TAVILY_API_KEY": "test-tavily",
    "SUPABASE_URL": "https://test.supabase.co",
    "SUPABASE_SERVICE_ROLE_KEY": "test-sb-key",
    "TELEGRAM_BOT_TOKEN": "test-bot",
    "TELEGRAM_CHAT_ID": "12345",
    "LINKEDIN_ACCESS_TOKEN": "test-li-token",
    "LINKEDIN_AUTHOR_URN": "urn:li:person:TEST",
    "LINKEDIN_TOKEN_EXPIRY_DATE": "2099-12-31",
    "X_API_KEY": "test-x-key",
    "X_API_SECRET": "test-x-secret",
    "X_ACCESS_TOKEN": "test-x-access",
    "X_ACCESS_SECRET": "test-x-access-secret",
}
for _key, _value in _REQUIRED_ENV.items():
    os.environ.setdefault(_key, _value)

import httpx  # noqa: E402
import tweepy  # noqa: E402
from src import publishers  # noqa: E402


def fake_settings():
    """Mutable config stand-in (real settings is a frozen dataclass)."""
    return SimpleNamespace(
        LINKEDIN_ACCESS_TOKEN="test-li-token",
        LINKEDIN_AUTHOR_URN="urn:li:person:TEST",
        X_API_KEY="test-x-key",
        X_API_SECRET="test-x-secret",
        X_ACCESS_TOKEN="test-x-access",
        X_ACCESS_SECRET="test-x-secret-token",
    )


class FakeLinkedInResponse:
    """Minimal httpx response stand-in with a headers dict (dict.get matches
    `_LI_*`'s use of `resp.headers.get(...)`)."""

    def __init__(self, status_code=201, headers=None, body=""):
        self.status_code = status_code
        # Real httpx.Headers: production code reads `resp.headers.get("x-restli-id")`
        # which is case-insensitive — a plain dict would not reproduce that.
        self.headers = httpx.Headers(headers or {})
        self._body = body

    def raise_for_status(self):
        if self.status_code >= 400:
            request = httpx.Request("POST", "https://api.linkedin.com/v2/ugcPosts")
            raise httpx.HTTPStatusError(
                "error",
                request=request,
                response=httpx.Response(self.status_code, request=request, text=self._body),
            )


class FakeHttpxClient:
    def __init__(self, response):
        self.response = response
        self.posted = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def post(self, url, headers=None, json=None):
        self.posted = (url, headers, json)
        return self.response


class FakeTweepyResponse:
    def __init__(self, tweet_id):
        self.data = {"id": tweet_id}


class FakeTweepyClient:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.text = None

    def create_tweet(self, text):
        self.text = text
        return FakeTweepyResponse("1793287460812324872")


class PublishToLinkedInTest(unittest.TestCase):
    def _publish(self, response):
        fake = FakeHttpxClient(response)
        with patch("src.publishers.httpx.Client", return_value=fake) as client_cls, \
             patch("src.publishers.settings", fake_settings()):
            result = publishers.publish_to_linkedin(
                "Sharing a thought about agentic AI.",
                image_bytes=None,
                article_url="https://example.com",
            )
        return result, fake, client_cls

    def test_returns_post_urn_from_x_restli_id_header(self):
        """Regression: LinkedIn returns the new post id in the x-restli-id
        RESPONSE HEADER (a URN). The old x-linkedin-id header is not an API
        contract and must not be read."""
        urn = "urn:li:ugcPost:6844785523593124080"
        result, fake, _ = self._publish(FakeLinkedInResponse(headers={"X-Restli-Id": urn}))
        self.assertEqual(result, urn)

    def test_post_id_not_from_x_linkedin_id_header(self):
        """A success response that ONLY carries x-linkedin-id (no x-restli-id)
        yields None — we do not read an undocumented header."""
        result, fake, _ = self._publish(
            FakeLinkedInResponse(headers={"X-Linkedin-Id": "12345"})
        )
        self.assertIsNone(result)

    def test_returns_none_when_post_id_header_is_absent(self):
        result, fake, _ = self._publish(FakeLinkedInResponse())
        self.assertIsNone(result)

    def test_raises_runtime_error_on_api_failure(self):
        with self.assertRaises(RuntimeError) as ctx:
            self._publish(FakeLinkedInResponse(status_code=429, body="rate limited"))
        self.assertIn("LinkedIn", str(ctx.exception))


class PublishToXTest(unittest.TestCase):
    def _publish(self, client_factory):
        # real config.settings is a frozen dataclass and may already be loaded
        # without X_* env, so swap the whole module-level settings object.
        with patch("src.publishers.tweepy.Client", side_effect=client_factory), \
             patch("src.publishers.settings", fake_settings()):
            return publishers.publish_to_x("Heads-up on orchestration.")

    def test_returns_tweet_id_from_response_body(self):
        result = self._publish(FakeTweepyClient)
        self.assertEqual(result, "1793287460812324872")

    def test_raises_runtime_error_on_tweepy_exception(self):
        def _boom(**kwargs):
            client = FakeTweepyClient(**kwargs)

            def _create_tweet(text):
                raise tweepy.TweepyException("rate limited")

            client.create_tweet = _create_tweet
            return client

        with self.assertRaises(RuntimeError) as ctx:
            self._publish(_boom)
        self.assertIn("X (Twitter)", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()