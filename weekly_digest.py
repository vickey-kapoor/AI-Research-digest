"""Weekly 'Best of the Week' Telegram digest — runs every Sunday."""

import os
import sys

from dotenv import load_dotenv

from src.kv_client import kv_get_list, kv_delete
from src.logger import get_logger
from src.telegram_sender import send_telegram_message, _escape_markdown, _truncate_message

logger = get_logger(__name__)

WEEKLY_KEY = "digest:weekly"
MIN_PAPERS = 3
MAX_PAPERS = 5


def _deduplicate(papers):
    """Deduplicate papers by title, keeping the first occurrence."""
    seen = set()
    unique = []
    for p in papers:
        title_key = p.get("title", "").strip().lower()
        if title_key not in seen:
            seen.add(title_key)
            unique.append(p)
    return unique


def _format_weekly_message(papers):
    """Format the weekly digest Telegram message (Markdown)."""
    count = len(papers)
    lines = [f"\U0001f5d3 *Week in AI Labs*\n"]
    lines.append(
        f"Here are the {count} most important lab developments from this week\\.\n"
    )

    for i, p in enumerate(papers, 1):
        title = _escape_markdown(p.get("title", "Untitled"))
        source = _escape_markdown(p.get("source", "Unknown"))
        item_type = p.get("type", "")
        if item_type:
            item_type = f" \u00b7 {_escape_markdown(item_type)}"
        topic = p.get("topic_id") or ""
        if topic:
            topic = f"#{_escape_markdown(topic)}"
        release_type = p.get("release_type", "")
        if release_type:
            release_type = _escape_markdown(release_type)
        # One-liner is what shipped, with why-it-matters as the italic kicker
        what_shipped = _escape_markdown(p.get("what_shipped", ""))
        why_it_matters = _escape_markdown(p.get("why_it_matters", ""))
        url = p.get("url", "")

        entry = f"{i}\\. *{title}*\n"
        entry += f"_{source}{item_type}_\n"
        meta = []
        if topic:
            meta.append(topic)
        if release_type:
            meta.append(release_type)
        if meta:
            entry += " \u00b7 ".join(meta) + "\n"
        if what_shipped:
            entry += f"{what_shipped}\n"
        if why_it_matters:
            entry += f"_{why_it_matters}_\n"
        if url:
            entry += f"{url}\n"
        lines.append(entry)

    lines.append("Full summaries were sent daily\\. See you next week\\.")
    return _truncate_message("\n".join(lines))


def main():
    """Build and send the weekly digest, then clear the KV list."""
    load_dotenv()

    telegram_token = os.getenv("TELEGRAM_BOT_TOKEN")
    telegram_chat_id = os.getenv("TELEGRAM_CHAT_ID")
    if not telegram_token or not telegram_chat_id:
        logger.error("Missing TELEGRAM_BOT_TOKEN or TELEGRAM_CHAT_ID")
        sys.exit(1)

    # Fetch the week's papers from KV
    logger.info("Fetching weekly papers from KV...")
    papers = kv_get_list(WEEKLY_KEY)

    if len(papers) < MIN_PAPERS:
        logger.info(
            "Only %d papers stored (need %d). Skipping weekly digest.",
            len(papers),
            MIN_PAPERS,
        )
        return

    # Deduplicate and take top N
    papers = _deduplicate(papers)[:MAX_PAPERS]
    logger.info("Sending weekly digest with %d papers", len(papers))

    # Format and send
    message = _format_weekly_message(papers)
    try:
        send_telegram_message(telegram_token, telegram_chat_id, message)
        logger.info("Weekly digest sent successfully")
    except Exception as e:
        logger.error("Failed to send weekly digest: %s", e)
        sys.exit(1)

    # Clear the list for next week
    try:
        kv_delete(WEEKLY_KEY)
        logger.info("Cleared weekly KV list")
    except Exception as e:
        logger.warning("Could not clear weekly KV list: %s", e)


if __name__ == "__main__":
    main()
