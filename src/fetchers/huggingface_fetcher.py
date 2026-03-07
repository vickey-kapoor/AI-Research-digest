"""Fetch AI research papers from Hugging Face Daily Papers."""

from datetime import datetime
from typing import Optional

import requests

from src.constants import FILTER_KEYWORDS, HF_DAILY_PAPERS_URL, REQUEST_TIMEOUT
from src.logger import get_logger
from src.utils.retry import retry_with_backoff

logger = get_logger(__name__)


@retry_with_backoff(
    max_retries=2,
    base_delay=1.0,
    exceptions=(requests.Timeout, requests.ConnectionError),
)
def _fetch_hf_data() -> list:
    """Fetch data from Hugging Face API with retry."""
    response = requests.get(HF_DAILY_PAPERS_URL, timeout=REQUEST_TIMEOUT)
    response.raise_for_status()
    return response.json()


def _matches_keywords(text: str) -> bool:
    """Check if text contains any agent/reasoning keywords."""
    text_lower = text.lower()
    return any(kw.lower() in text_lower for kw in FILTER_KEYWORDS)


def fetch_huggingface_papers(max_results: int = 5) -> list[dict]:
    """
    Fetch recent AI Agents & Reasoning papers from Hugging Face Daily Papers.

    Args:
        max_results: Maximum number of papers to return

    Returns:
        List of normalized paper dictionaries
    """
    try:
        data = _fetch_hf_data()

        papers = []
        for item in data:
            paper = item.get("paper", {})
            title = paper.get("title", "")
            abstract = paper.get("summary", "")

            # Filter for agent/reasoning relevance
            if not _matches_keywords(title) and not _matches_keywords(abstract):
                continue

            # Extract authors
            authors_list = paper.get("authors", [])
            authors = ", ".join(
                a.get("name", "Unknown") for a in authors_list[:5]
            )
            if len(authors_list) > 5:
                authors += f" et al. ({len(authors_list)} authors)"

            # Parse published date
            published = paper.get("publishedAt", "")
            try:
                published_at = datetime.fromisoformat(
                    published.replace("Z", "+00:00")
                ).isoformat()
            except (ValueError, AttributeError):
                published_at = datetime.now().isoformat()

            # Clean abstract
            if len(abstract) > 500:
                abstract = abstract[:497] + "..."

            # Validate paper ID before constructing URL
            paper_id = paper.get('id', '')
            if paper_id and isinstance(paper_id, str):
                paper_url = f"https://huggingface.co/papers/{paper_id}"
            else:
                paper_url = "https://huggingface.co/papers"

            paper_dict = {
                "title": title.strip(),
                "description": abstract.replace("\n", " ").strip(),
                "source": "Hugging Face",
                "url": paper_url,
                "published_at": published_at,
                "type": "research",
                "authors": authors,
                "topics": ["AI Agents", "Reasoning"],
            }
            papers.append(paper_dict)

            if len(papers) >= max_results:
                break

        return papers

    except requests.Timeout:
        logger.error("Hugging Face request timed out")
        return []
    except requests.RequestException:
        logger.error("Failed to fetch Hugging Face papers")
        return []
    except Exception:
        logger.error("Failed to parse Hugging Face response")
        return []
