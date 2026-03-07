"""Fetch AI research papers from arXiv."""

from datetime import datetime, timedelta
from typing import Optional
import urllib.parse
import socket

import feedparser

from src.constants import (
    ARXIV_API_URL,
    ARXIV_CATEGORIES,
    AGENT_REASONING_KEYWORDS,
    REQUEST_TIMEOUT,
)
from src.logger import get_logger
from src.utils.retry import retry_with_backoff

logger = get_logger(__name__)


@retry_with_backoff(
    max_retries=2,
    base_delay=1.0,
    exceptions=(socket.timeout, OSError),
)
def _parse_arxiv_feed(url: str):
    """Parse arXiv feed with retry on timeout."""
    old_timeout = socket.getdefaulttimeout()
    socket.setdefaulttimeout(REQUEST_TIMEOUT)
    try:
        return feedparser.parse(url)
    finally:
        socket.setdefaulttimeout(old_timeout)


def fetch_arxiv_papers(max_results: int = 5) -> list[dict]:
    """
    Fetch recent AI Agents & Reasoning papers from arXiv.

    Args:
        max_results: Maximum number of papers to return

    Returns:
        List of normalized paper dictionaries
    """
    # Build search query for AI/ML categories with agent/reasoning focus
    categories = "(" + " OR ".join(f"cat:{cat}" for cat in ARXIV_CATEGORIES) + ")"
    keywords = " OR ".join(f'all:"{kw}"' for kw in AGENT_REASONING_KEYWORDS[:5])
    query = f"{categories} AND ({keywords})"

    params = {
        "search_query": query,
        "start": 0,
        "max_results": max_results * 2,  # Fetch extra to filter
        "sortBy": "submittedDate",
        "sortOrder": "descending",
    }

    url = f"{ARXIV_API_URL}?{urllib.parse.urlencode(params)}"

    try:
        feed = _parse_arxiv_feed(url)

        if feed.bozo and not feed.entries:
            logger.warning("arXiv feed parse error")
            return []

        papers = []
        for entry in feed.entries[:max_results]:
            # Extract authors
            authors = ", ".join(
                author.get("name", "Unknown") for author in entry.get("authors", [])
            )
            if len(authors) > 100:
                authors = authors[:97] + "..."

            # Parse published date
            published = entry.get("published", "")
            try:
                published_dt = datetime.strptime(published[:10], "%Y-%m-%d")
                published_at = published_dt.isoformat()
            except (ValueError, IndexError):
                published_at = datetime.now().isoformat()

            # Extract categories/topics
            categories = [tag.get("term", "") for tag in entry.get("tags", [])]
            topics = [c for c in categories if c.startswith("cs.")]

            # Clean abstract
            abstract = entry.get("summary", "").replace("\n", " ").strip()
            if len(abstract) > 500:
                abstract = abstract[:497] + "..."

            paper = {
                "title": entry.get("title", "").replace("\n", " ").strip(),
                "description": abstract,
                "source": "arXiv",
                "url": entry.get("link", ""),
                "published_at": published_at,
                "type": "research",
                "authors": authors,
                "topics": topics,
            }
            papers.append(paper)

        return papers

    except socket.timeout:
        logger.error("arXiv request timed out")
        return []
    except Exception:
        logger.error("Failed to fetch arXiv papers")
        return []
