"""
PersonaPulse – Token Canary Check
===================================
Standalone utility that checks LinkedIn OAuth token health
and fires an urgent Telegram alert if expiry is imminent.

Can be run independently as a pre-flight check or called
from the LangGraph agent as the first node.

Usage (CLI):
    python -m src.canary
"""

from __future__ import annotations

import logging
import sys

from src.llm import send_telegram_text
from src.publishers import check_linkedin_token_health

log = logging.getLogger(__name__)


def run_canary_check() -> bool:
    """
    Perform the LinkedIn token canary check.

    Returns
    -------
    True  – token is healthy, pipeline may continue.
    False – token is critical/expired; urgent alert sent, pipeline must STOP.
    """
    health = check_linkedin_token_health()

    if health["is_expired"]:
        message = (
            "🚨 *URGENT – LinkedIn Token EXPIRED!*\n\n"
            f"Your LinkedIn access token expired on *{health['expires_on']}*.\n"
            "Posts cannot be published until you renew the token.\n\n"
            "➡️ Renew at: https://www.linkedin.com/developers/apps\n"
            "_PersonaPulse pipeline halted._"
        )
        send_telegram_text(message)
        log.error("[Canary] Token expired – pipeline halted.")
        return False

    if health["is_critical"]:
        days = health["days_remaining"]
        message = (
            f"⚠️ *LinkedIn Token Expiring Soon!*\n\n"
            f"Your token expires in *{days} day{'s' if days != 1 else ''}* "
            f"(on {health['expires_on']}).\n"
            "Please renew it before it expires to avoid disruption.\n\n"
            "➡️ Renew at: https://www.linkedin.com/developers/apps\n"
            "_PersonaPulse will continue this run but may fail soon._"
        )
        send_telegram_text(message)
        log.warning("[Canary] Token critical (%d days left) – alert sent.", days)
        # Pipeline continues but with warning
        return True

    log.info("[Canary] Token healthy: %d days remaining.", health["days_remaining"])
    return True


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s │ %(message)s")
    ok = run_canary_check()
    sys.exit(0 if ok else 1)
