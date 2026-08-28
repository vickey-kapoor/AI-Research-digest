"""Unit tests for blog fetcher module."""

import socket
from unittest.mock import patch, MagicMock

import pytest

from src.constants import BLOG_FEEDS
from src.fetchers.blog_fetcher import (
    fetch_blog_posts,
    _fetch_single_feed,
    _is_dev_relevant,
    _is_tutorial,
)


class TestBlogFetcher:
    """Tests for blog fetcher."""

    def test_fetch_single_feed_success(self, mock_blog_feed):
        """Test successful single feed fetching."""
        with patch("src.fetchers.blog_fetcher.feedparser.parse") as mock_parse:
            mock_response = MagicMock()
            mock_response.bozo = False
            mock_response.entries = mock_blog_feed["entries"]
            mock_parse.return_value = mock_response

            posts = _fetch_single_feed("OpenAI", "https://openai.com/blog/rss.xml", 5)

            assert len(posts) == 1
            assert posts[0]["source"] == "OpenAI"
            assert posts[0]["type"] == "announcement"

    def test_fetch_single_feed_timeout(self):
        """Test single feed handles timeout gracefully."""
        with patch("src.fetchers.blog_fetcher.feedparser.parse") as mock_parse:
            mock_parse.side_effect = socket.timeout()

            posts = _fetch_single_feed("Test Blog", "https://test.com/rss", 5)

            assert posts == []

    def test_fetch_blog_posts_aggregates_sources(self, mock_blog_feed):
        """Test that blog posts are aggregated from multiple sources."""
        with patch("src.fetchers.blog_fetcher._fetch_single_feed") as mock_fetch:
            mock_fetch.return_value = [
                {
                    "title": "Test Post",
                    "summary": "Test description about new API launch",
                    "url": "https://test.com/post",
                    "source": "Test Blog",
                    "published": "2024-01-15T00:00:00",
                    "type": "announcement",
                }
            ]

            posts = fetch_blog_posts(max_results=5)

            # Should be called once per configured blog feed
            assert mock_fetch.call_count == len(BLOG_FEEDS)

    def test_fetch_single_feed_parse_error(self):
        """Test single feed handles parse errors gracefully."""
        with patch("src.fetchers.blog_fetcher.feedparser.parse") as mock_parse:
            mock_response = MagicMock()
            mock_response.bozo = True
            mock_response.entries = []
            mock_parse.return_value = mock_response

            posts = _fetch_single_feed("Test Blog", "https://test.com/rss", 5)

            assert posts == []

    def test_fetch_single_feed_respects_max(self, mock_blog_feed):
        """Test that single feed respects max_per_source limit."""
        with patch("src.fetchers.blog_fetcher.feedparser.parse") as mock_parse:
            mock_response = MagicMock()
            mock_response.bozo = False
            # Create multiple entries
            mock_response.entries = mock_blog_feed["entries"] * 10
            mock_parse.return_value = mock_response

            posts = _fetch_single_feed("OpenAI", "https://openai.com/blog/rss.xml", 2)

            assert len(posts) <= 2


class TestDevRelevanceFilter:
    """Tests for _is_dev_relevant filtering (AI lab developments)."""

    def test_is_dev_relevant_accepts_lab_release(self):
        """Post announcing a lab model release is accepted."""
        post = {"title": "Introducing our new frontier model", "summary": "A reasoning model now available in the developer api"}
        assert _is_dev_relevant(post) is True

    def test_is_dev_relevant_rejects_corporate_news(self):
        """Post about hiring is rejected."""
        post = {"title": "We are hiring across the company", "summary": "Open roles on every team"}
        assert _is_dev_relevant(post) is False

    def test_exclude_takes_precedence_over_include(self):
        """Post with both include and exclude keywords is rejected."""
        post = {"title": "New model release as the company raises a funding round", "summary": "Frontier model shipped alongside a series b"}
        assert _is_dev_relevant(post) is False

    def test_no_keyword_match_rejected(self):
        """Generic post with no matching keywords is rejected."""
        post = {"title": "Our company vision for the future", "summary": "Thoughts on progress"}
        assert _is_dev_relevant(post) is False

    def test_filter_fallback_when_all_filtered(self, mock_blog_feed):
        """When all posts are filtered out, unfiltered list is returned."""
        with patch("src.fetchers.blog_fetcher.feedparser.parse") as mock_parse:
            mock_response = MagicMock()
            mock_response.bozo = False
            # Create entries that won't match any filter keywords
            entry = MagicMock()
            entry.get = lambda k, d="": {
                "title": "Our company culture",
                "summary": "A day in the life",
                "summary": "",
                "link": "https://test.com/post",
            }.get(k, d)
            entry.__contains__ = lambda self, k: k in ["title", "summary"]
            mock_response.entries = [entry]
            mock_parse.return_value = mock_response

            posts = _fetch_single_feed("TestBlog", "https://test.com/rss", 5)

            # Should return unfiltered posts as fallback
            assert len(posts) == 1
            assert posts[0]["title"] == "Our company culture"

    def test_no_fallback_with_explicit_keywords(self, mock_blog_feed):
        """An explicit topic keyword list is respected strictly — no fallback."""
        with patch("src.fetchers.blog_fetcher.feedparser.parse") as mock_parse:
            mock_response = MagicMock()
            mock_response.bozo = False
            entry = MagicMock()
            entry.get = lambda k, d="": {
                "title": "Our company culture",
                "summary": "",
                "link": "https://test.com/post",
            }.get(k, d)
            entry.__contains__ = lambda self, k: k in ["title", "summary"]
            mock_response.entries = [entry]
            mock_parse.return_value = mock_response

            posts = _fetch_single_feed(
                "TestBlog", "https://test.com/rss", 5, filter_keywords=["frontier model"]
            )

            assert posts == []


class TestTutorialFilter:
    """Tests for excluding vendor how-tos from the candidate pool."""

    def test_blocks_the_real_winners(self):
        """Both items that actually won the daily pick were tutorials."""
        assert _is_tutorial("Preparing data for supervised fine-tuning Part 2: Advanced data strategies")
        assert _is_tutorial("Build agentic creative workflows with Amazon Quick and fal")

    @pytest.mark.parametrize("title", [
        "Gemini-3.5-Transcribe",
        "Gemini Omni 1.1 Flash",
        "Piloting the world's first double-blind AI evaluations",
        "The Hugging Face incident and the road ahead",
        "Qwen3.8-Flash-Next: A New Architecture, Towards Ultimate Cost Efficiency",
        "GSPO: Towards Scalable Reinforcement Learning for Language Models",
        "Introducing GPT-5.6 in the API",
        "anthropic-sdk-python v1.2.0 released",
    ])
    def test_real_lab_news_survives(self, title):
        """False positives here would silently drop genuine launches."""
        assert not _is_tutorial(title)

    def test_matches_title_only_not_summary(self):
        """A launch post whose body says 'how to' must not be dropped."""
        post = {
            "title": "Introducing our new frontier model",
            "summary": "We show how to call the new developer api endpoint.",
        }
        assert _is_dev_relevant(post) is True
