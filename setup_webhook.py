#!/usr/bin/env python3
"""
PersonaPulse – Telegram Webhook Setup Script
===========================================
Registers your Supabase Edge Function URL as the Telegram webhook with
two-layer security: passes a secret_token for X-Telegram-Bot-Api-Secret-Token validation.

Usage:
    python setup_webhook.py                     # Auto-detects URL and token from .env
    python setup_webhook.py --info              # Check current webhook status
    python setup_webhook.py --delete            # Remove registered webhook
    python setup_webhook.py --url <CUSTOM_URL>  # Override webhook endpoint URL
"""

import argparse
import hashlib
import json
import os
import re
import sys
from pathlib import Path

try:
    import requests
    from dotenv import load_dotenv
except ImportError:
    print("[Error] Missing dependencies. Run: pip install requests python-dotenv")
    sys.exit(1)

# Load environment variables from .env in workspace root
ROOT_DIR = Path(__file__).resolve().parent
load_dotenv(ROOT_DIR / ".env")

SECRET_TOKEN_PATTERN = re.compile(r"^[a-zA-Z0-9_-]{1,256}$")


def derive_secret_token(bot_token: str) -> str:
    """Derive a valid Telegram secret token [a-zA-Z0-9_-]{1,256} from the bot token."""
    return hashlib.sha256(bot_token.strip().encode("utf-8")).hexdigest()


def get_webhook_secret(bot_token: str, custom_secret: str | None = None) -> str:
    """
    Returns the webhook secret:
    1. CLI arg or TELEGRAM_WEBHOOK_SECRET env var if specified.
    2. Fallback: SHA-256 hex digest of TELEGRAM_BOT_TOKEN (64 chars, 0-9 and a-f).
    """
    secret = (custom_secret or os.getenv("TELEGRAM_WEBHOOK_SECRET") or "").strip()
    if secret:
        if not SECRET_TOKEN_PATTERN.match(secret):
            raise ValueError(
                f"Secret token '{secret}' is invalid. Telegram requires characters matching [a-zA-Z0-9_-]{{1,256}}."
            )
        return secret
    return derive_secret_token(bot_token)


def get_default_webhook_url() -> str | None:
    """Constructs the default Supabase Edge Function webhook URL if SUPABASE_URL is configured."""
    explicit_url = os.getenv("TELEGRAM_WEBHOOK_URL")
    if explicit_url:
        return explicit_url.strip()

    supabase_url = os.getenv("SUPABASE_URL", "").strip()
    if supabase_url:
        # e.g., https://abcdefghijklm.supabase.co -> https://abcdefghijklm.supabase.co/functions/v1/telegram-webhook
        base = supabase_url.rstrip("/")
        return f"{base}/functions/v1/telegram-webhook"
    return None


def get_webhook_info(bot_token: str) -> None:
    """Fetches and displays current webhook information from Telegram."""
    api_url = f"https://api.telegram.org/bot{bot_token}/getWebhookInfo"
    resp = requests.get(api_url, timeout=10)
    data = resp.json()
    print("\n📡 Current Webhook Info:")
    print(json.dumps(data, indent=2))


def delete_webhook(bot_token: str) -> None:
    """Deletes the current Telegram webhook."""
    api_url = f"https://api.telegram.org/bot{bot_token}/deleteWebhook"
    resp = requests.post(api_url, json={"drop_pending_updates": True}, timeout=10)
    data = resp.json()
    if data.get("ok"):
        print("✅ Webhook deleted successfully.")
    else:
        print(f"❌ Failed to delete webhook: {data}")


def set_webhook(bot_token: str, webhook_url: str, secret_token: str, drop_pending: bool = False) -> None:
    """Registers the Telegram webhook with secret_token."""
    api_url = f"https://api.telegram.org/bot{bot_token}/setWebhook"
    payload = {
        "url": webhook_url,
        "secret_token": secret_token,
        "allowed_updates": ["message", "callback_query"],
        "drop_pending_updates": drop_pending,
    }

    print(f"🔒 Setting Telegram Webhook...")
    print(f"   Endpoint:     {webhook_url}")
    print(f"   Secret Token: {secret_token[:8]}...{secret_token[-8:]} (Length: {len(secret_token)})")

    resp = requests.post(api_url, json=payload, timeout=15)
    data = resp.json()

    if data.get("ok"):
        print(f"\n✅ Webhook registered successfully!")
        print(f"   Telegram description: {data.get('description', 'OK')}")
        print("\n💡 Important Reminder:")
        print("   Make sure the matching secrets are configured in Supabase Edge Functions:")
        print(f'   supabase secrets set TELEGRAM_BOT_TOKEN="{bot_token[:10]}..." \\')
        print(f'     TELEGRAM_CHAT_ID="{os.getenv("TELEGRAM_CHAT_ID", "<YOUR_CHAT_ID>")}" \\')
        if os.getenv("TELEGRAM_WEBHOOK_SECRET"):
            print(f'     TELEGRAM_WEBHOOK_SECRET="{secret_token}"')
        else:
            print(f'     # (TELEGRAM_WEBHOOK_SECRET is automatically derived from TELEGRAM_BOT_TOKEN via SHA-256)')
    else:
        print(f"\n❌ Error setting webhook:")
        print(json.dumps(data, indent=2))
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description="Configure Telegram Webhook with two-layer security.")
    parser.add_argument("--url", help="Webhook endpoint URL (defaults to SUPABASE_URL/functions/v1/telegram-webhook)")
    parser.add_argument("--secret", help="Custom secret_token ([a-zA-Z0-9_-]{1,256})")
    parser.add_argument("--drop-pending", action="store_true", help="Drop pending Telegram updates")
    parser.add_argument("--info", action="store_true", help="Query getWebhookInfo from Telegram and exit")
    parser.add_argument("--delete", action="store_true", help="Delete Telegram webhook and exit")
    args = parser.parse_args()

    bot_token = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
    if not bot_token:
        print("[Error] Missing TELEGRAM_BOT_TOKEN in environment or .env file.")
        sys.exit(1)

    if args.info:
        get_webhook_info(bot_token)
        return

    if args.delete:
        delete_webhook(bot_token)
        return

    webhook_url = args.url or get_default_webhook_url()
    if not webhook_url:
        print("[Error] Webhook URL not provided. Set SUPABASE_URL in .env or provide --url <URL>.")
        sys.exit(1)

    secret_token = get_webhook_secret(bot_token, args.secret)
    set_webhook(bot_token, webhook_url, secret_token, drop_pending=args.drop_pending)
    get_webhook_info(bot_token)


if __name__ == "__main__":
    main()
