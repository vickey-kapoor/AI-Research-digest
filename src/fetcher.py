"""Aggregate items from all fetcher sources (blogs, GitHub, Hacker News)."""

from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta, timezone

from src.constants import (
    DIGEST_MAX_AGE_HOURS,
    MAX_ITEMS_PER_SOURCE,
    THREAD_POOL_WORKERS,
)
from src.fetchers.blog_fetcher import fetch_blog_posts
from src.fetchers.github_fetcher import fetch_github_releases
from src.fetchers.hackernews_fetcher import fetch_hackernews_stories
from src.fetchers.huggingface_fetcher import fetch_huggingface_papers
from src.logger import get_logger
from src.topic_config import get_active_topics

logger = get_logger(__name__)

# Blog candidates pulled before ranking — kept above DIGEST_MAX_RESULTS so a
# dozen lab feeds each get a real shot at the daily pick.
BLOG_MAX_RESULTS = 25


def _parse_published(value: str) -> datetime | None:
    """Parse a published timestamp from any source into aware UTC.

    The four fetchers emit slightly different shapes — trailing "Z", explicit
    offsets, and (historically) naive strings already stored in papers.json.
    A naive value is treated as UTC, which is what every source actually
    produces. Returns None when the value cannot be parsed at all.
    """
    if not value or not isinstance(value, str):
        return None
    try:
        parsed = datetime.fromisoformat(value.strip().replace("Z", "+00:00"))
    except (ValueError, TypeError):
        return None
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _filter_by_recency(items: list[dict], max_age_hours: int) -> list[dict]:
    """Drop items published longer than max_age_hours ago.

    Blog feeds carry no cutoff of their own, so without this a quiet day can
    surface a week-old post as today's development. Items with an unparseable
    timestamp are kept: failing open costs a stale item, while failing closed
    could silently empty the digest.
    """
    if max_age_hours <= 0:
        return items

    cutoff = datetime.now(timezone.utc) - timedelta(hours=max_age_hours)
    kept, stale, undated = [], 0, 0

    for item in items:
        published = _parse_published(item.get("published", ""))
        if published is None:
            undated += 1
            kept.append(item)
            continue
        if published < cutoff:
            stale += 1
            continue
        kept.append(item)

    if stale:
        logger.info("Dropped %d item(s) older than %dh", stale, max_age_hours)
    if undated:
        logger.warning("Kept %d item(s) with an unparseable published date", undated)

    return kept


def _cap_per_source(items: list[dict], max_per_source: int) -> list[dict]:
    """Limit how many slots any one source can occupy.

    Items must already be sorted newest-first: each source keeps its most
    recent entries. Without this a high-volume vendor blog crowds out the
    frontier labs — one held 4 of 10 slots on 28-Aug and won the daily pick.
    """
    if max_per_source <= 0:
        return items

    counts: dict[str, int] = {}
    kept, dropped = [], 0

    for item in items:
        source = item.get("source", "")
        seen = counts.get(source, 0)
        if seen >= max_per_source:
            dropped += 1
            continue
        counts[source] = seen + 1
        kept.append(item)

    if dropped:
        logger.info("Dropped %d item(s) over the %d-per-source cap", dropped, max_per_source)

    return kept


def _deduplicate_by_url(items: list[dict]) -> list[dict]:
    """Remove duplicate items based on URL."""
    seen_urls: set[str] = set()
    unique = []
    for item in items:
        url = item.get("url", "")
        if url and url in seen_urls:
            continue
        if url:
            seen_urls.add(url)
        unique.append(item)
    return unique


def fetch_all(max_results: int = 20, filter_keywords: list[str] | None = None) -> list[dict]:
    """Fetch items from all sources in parallel, deduplicate, and sort by date.

    Args:
        max_results: Maximum number of items to return
        filter_keywords: Keyword list used to filter every source to matching topics only

    Returns:
        Combined, deduplicated, keyword-filtered, and sorted list of items
    """
    fetchers = [
        ("Blogs", lambda: fetch_blog_posts(max_results=BLOG_MAX_RESULTS, filter_keywords=filter_keywords)),
        ("GitHub", fetch_github_releases),
        ("Hacker News", lambda: fetch_hackernews_stories(filter_keywords=filter_keywords)),
        ("HF Papers", lambda: fetch_huggingface_papers(filter_keywords=filter_keywords)),
    ]

    all_items: list[dict] = []

    with ThreadPoolExecutor(max_workers=THREAD_POOL_WORKERS) as executor:
        future_to_source = {
            executor.submit(fn): name
            for name, fn in fetchers
        }

        for future in as_completed(future_to_source):
            source = future_to_source[future]
            try:
                results = future.result()
                logger.info("Fetched %d items from %s", len(results), source)
                all_items.extend(results)
            except Exception:
                logger.error("Failed to fetch from %s", source)

    # Drop anything too old to be "today's" development. Applied centrally so
    # every source gets the same treatment, not just the blogs that lack a
    # cutoff — this also backstops a per-source cutoff regressing.
    all_items = _filter_by_recency(all_items, DIGEST_MAX_AGE_HOURS)
    logger.info("After recency filter: %d items within %dh", len(all_items), DIGEST_MAX_AGE_HOURS)

    # Deduplicate by URL
    unique = _deduplicate_by_url(all_items)
    logger.info("After deduplication: %d unique items", len(unique))

    # Final keyword filter — drop anything that doesn't mention a topic keyword
    # Also assign topic_id to each item based on first matching topic
    if filter_keywords:
        topics = get_active_topics()
        kw_lower = [k.lower() for k in filter_keywords]
        filtered = []
        for item in unique:
            text = (item.get("title", "") + " " + item.get("summary", "")).lower()
            if not any(kw in text for kw in kw_lower):
                continue
            # Tag with the first topic whose keywords match
            if not item.get("topic_id"):
                for topic in topics:
                    if any(kw.lower() in text for kw in topic["keywords"]):
                        item = {**item, "topic_id": topic["id"]}
                        break
            filtered.append(item)
        unique = filtered
        logger.info("After keyword filter: %d items match active topics", len(unique))

    # Sort by published date (most recent first)
    unique.sort(key=lambda x: x.get("published", ""), reverse=True)

    # Cap per source after sorting so each source keeps its freshest items
    unique = _cap_per_source(unique, MAX_ITEMS_PER_SOURCE)
    logger.info("After per-source cap: %d items", len(unique))

    return unique[:max_results]
