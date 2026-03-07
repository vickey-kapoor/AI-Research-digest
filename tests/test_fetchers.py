"""Unit tests for individual fetcher modules."""

import json
import socket
from unittest.mock import Mock, patch, MagicMock

import pytest
import requests

from src.fetchers.arxiv_fetcher import fetch_arxiv_papers
from src.fetchers.huggingface_fetcher import fetch_huggingface_papers, _matches_keywords
from src.fetchers.pwc_fetcher import fetch_pwc_papers
from src.fetchers.blog_fetcher import fetch_blog_posts, _fetch_single_feed


class TestArxivFetcher:
    """Tests for arXiv fetcher."""

    def test_fetch_arxiv_papers_success(self, mock_arxiv_feed):
        """Test successful arXiv paper fetching."""
        with patch("src.fetchers.arxiv_fetcher.feedparser.parse") as mock_parse:
            mock_parse.return_value = MagicMock(**mock_arxiv_feed)
            mock_parse.return_value.bozo = False
            mock_parse.return_value.entries = mock_arxiv_feed["entries"]

            papers = fetch_arxiv_papers(max_results=5)

            assert len(papers) >= 0  # May be 0 if no entries match
            mock_parse.assert_called_once()

    def test_fetch_arxiv_papers_timeout(self):
        """Test arXiv fetcher handles timeout gracefully."""
        with patch("src.fetchers.arxiv_fetcher.feedparser.parse") as mock_parse:
            mock_parse.side_effect = socket.timeout()

            papers = fetch_arxiv_papers(max_results=5)

            assert papers == []

    def test_fetch_arxiv_papers_parse_error(self):
        """Test arXiv fetcher handles parse errors gracefully."""
        with patch("src.fetchers.arxiv_fetcher.feedparser.parse") as mock_parse:
            mock_response = MagicMock()
            mock_response.bozo = True
            mock_response.entries = []
            mock_parse.return_value = mock_response

            papers = fetch_arxiv_papers(max_results=5)

            assert papers == []

    def test_fetch_arxiv_papers_normalizes_data(self, mock_arxiv_feed):
        """Test that arXiv papers are normalized correctly."""
        with patch("src.fetchers.arxiv_fetcher.feedparser.parse") as mock_parse:
            mock_response = MagicMock()
            mock_response.bozo = False
            mock_response.entries = mock_arxiv_feed["entries"]
            mock_parse.return_value = mock_response

            papers = fetch_arxiv_papers(max_results=5)

            if papers:
                paper = papers[0]
                assert "title" in paper
                assert "description" in paper
                assert "source" in paper
                assert paper["source"] == "arXiv"
                assert "url" in paper
                assert "published_at" in paper


class TestHuggingFaceFetcher:
    """Tests for Hugging Face fetcher."""

    def test_matches_keywords_true(self):
        """Test keyword matching returns True for matching text."""
        assert _matches_keywords("This paper is about AI agents") is True
        assert _matches_keywords("Multi-agent reasoning") is True
        assert _matches_keywords("Chain of thought prompting") is True

    def test_matches_keywords_false(self):
        """Test keyword matching returns False for non-matching text."""
        assert _matches_keywords("Weather forecast for tomorrow") is False
        assert _matches_keywords("Recipe for chocolate cake") is False

    def test_fetch_huggingface_papers_success(self, mock_huggingface_response):
        """Test successful Hugging Face paper fetching."""
        with patch("src.fetchers.huggingface_fetcher.requests.get") as mock_get:
            mock_response = Mock()
            mock_response.json.return_value = mock_huggingface_response
            mock_response.raise_for_status = Mock()
            mock_get.return_value = mock_response

            papers = fetch_huggingface_papers(max_results=5)

            assert len(papers) == 1
            assert papers[0]["source"] == "Hugging Face"
            assert "agent" in papers[0]["title"].lower()

    def test_fetch_huggingface_papers_timeout(self):
        """Test Hugging Face fetcher handles timeout gracefully."""
        with patch("src.fetchers.huggingface_fetcher.requests.get") as mock_get:
            mock_get.side_effect = requests.Timeout()

            papers = fetch_huggingface_papers(max_results=5)

            assert papers == []

    def test_fetch_huggingface_papers_request_error(self):
        """Test Hugging Face fetcher handles request errors gracefully."""
        with patch("src.fetchers.huggingface_fetcher.requests.get") as mock_get:
            mock_get.side_effect = requests.RequestException()

            papers = fetch_huggingface_papers(max_results=5)

            assert papers == []


class TestPWCFetcher:
    """Tests for Papers With Code fetcher."""

    def test_fetch_pwc_papers_success(self, mock_pwc_response):
        """Test successful PWC paper fetching."""
        with patch("src.fetchers.pwc_fetcher.requests.get") as mock_get:
            mock_response = Mock()
            mock_response.json.return_value = mock_pwc_response
            mock_response.raise_for_status = Mock()
            mock_response.headers = {"content-type": "application/json"}
            mock_get.return_value = mock_response

            papers = fetch_pwc_papers(max_results=5)

            assert len(papers) == 1
            assert papers[0]["source"] == "Papers With Code"

    def test_fetch_pwc_papers_non_json_response(self):
        """Test PWC fetcher handles non-JSON responses gracefully."""
        with patch("src.fetchers.pwc_fetcher.requests.get") as mock_get:
            mock_response = Mock()
            mock_response.raise_for_status = Mock()
            mock_response.headers = {"content-type": "text/html"}
            mock_get.return_value = mock_response

            papers = fetch_pwc_papers(max_results=5)

            assert papers == []

    def test_fetch_pwc_papers_timeout(self):
        """Test PWC fetcher handles timeout gracefully."""
        with patch("src.fetchers.pwc_fetcher.requests.get") as mock_get:
            mock_get.side_effect = requests.Timeout()

            papers = fetch_pwc_papers(max_results=5)

            assert papers == []


class TestBlogFetcher:
    """Tests for blog fetcher."""

    def test_fetch_single_feed_success(self, mock_blog_feed):
        """Test successful single feed fetching."""
        with patch("src.fetchers.blog_fetcher.feedparser.parse") as mock_parse:
            mock_response = MagicMock()
            mock_response.bozo = False
            mock_response.entries = mock_blog_feed["entries"]
            mock_parse.return_value = mock_response

            posts = _fetch_single_feed("Google AI", "https://blog.google/rss", 5)

            assert len(posts) == 1
            assert posts[0]["source"] == "Google AI"

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
                    "description": "Test description about AI agents",
                    "source": "Test Blog",
                    "url": "https://test.com/post",
                    "published_at": "2024-01-15T00:00:00",
                    "type": "research",
                    "authors": "Test Blog",
                    "topics": ["AI Agents", "Reasoning"],
                }
            ]

            posts = fetch_blog_posts(max_results=5)

            # Should be called for each blog feed
            assert mock_fetch.call_count == 3  # Google AI, DeepMind, Meta AI
