"""Fetch AI research papers from Papers With Code.

Note: The PWC API may require authentication or have rate limits.
This fetcher returns an empty list if the API is unavailable.
"""

from datetime import datetime
from typing import Optional, Tuple

import requests

from src.constants import FILTER_KEYWORDS, PWC_API_URL, REQUEST_TIMEOUT
from src.logger import get_logger
from src.utils.retry import retry_with_backoff

logger = get_logger(__name__)


@retry_with_backoff(
    max_retries=2,
    base_delay=1.0,
    exceptions=(requests.Timeout, requests.ConnectionError),
)
def _fetch_pwc_data(params: dict) -> Tuple[dict, str]:
    """Fetch data from Papers With Code API with retry."""
    response = requests.get(PWC_API_URL, params=params, timeout=REQUEST_TIMEOUT)
    response.raise_for_status()
    content_type = response.headers.get("content-type", "")
    return response.json(), content_type


def _matches_keywords(text: str) -> bool:
    """Check if text contains any agent/reasoning keywords."""
    text_lower = text.lower()
    return any(kw.lower() in text_lower for kw in FILTER_KEYWORDS)


def fetch_pwc_papers(max_results: int = 5) -> list[dict]:
    """
    Fetch recent AI Agents & Reasoning papers from Papers With Code.

    Args:
        max_results: Maximum number of papers to return

    Returns:
        List of normalized paper dictionaries
    """
    try:
        # Fetch recent papers, sorted by date
        params = {
            "ordering": "-published",
            "page_size": max_results * 4,  # Fetch extra to filter
        }
        data, content_type = _fetch_pwc_data(params)

        # Check if response is JSON
        if "application/json" not in content_type:
            # API might require auth or be temporarily unavailable
            return []

        papers = []
        results = data.get("results", [])

        for item in results:
            title = item.get("title", "")
            abstract = item.get("abstract", "")

            # Filter for agent/reasoning relevance
            if not _matches_keywords(title) and not _matches_keywords(abstract):
                continue

            # Extract authors
            authors = item.get("authors", [])
            if isinstance(authors, list):
                authors_str = ", ".join(authors[:5])
                if len(authors) > 5:
                    authors_str += f" et al. ({len(authors)} authors)"
            else:
                authors_str = str(authors) if authors else "Unknown"

            # Parse published date
            published = item.get("published", "")
            try:
                published_dt = datetime.strptime(published, "%Y-%m-%d")
                published_at = published_dt.isoformat()
            except (ValueError, TypeError):
                published_at = datetime.now().isoformat()

            # Clean abstract
            if len(abstract) > 500:
                abstract = abstract[:497] + "..."

            # Build URL
            paper_url = item.get("url_abs", "") or item.get("paper_url", "")
            if not paper_url and item.get("id"):
                paper_url = f"https://paperswithcode.com/paper/{item['id']}"

            paper_dict = {
                "title": title.strip(),
                "description": abstract.replace("\n", " ").strip(),
                "source": "Papers With Code",
                "url": paper_url,
                "published_at": published_at,
                "type": "research",
                "authors": authors_str,
                "topics": ["AI Agents", "Reasoning"],
            }
            papers.append(paper_dict)

            if len(papers) >= max_results:
                break

        return papers

    except requests.Timeout:
        logger.error("Papers With Code request timed out")
        return []
    except requests.RequestException:
        logger.error("Failed to fetch Papers With Code")
        return []
    except Exception:
        logger.error("Failed to parse Papers With Code response")
        return []
