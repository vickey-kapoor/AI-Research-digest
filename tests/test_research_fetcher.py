"""Unit tests for research fetcher module (aggregation and deduplication)."""

import pytest
from unittest.mock import patch, MagicMock

from src.research_fetcher import (
    fetch_ai_research,
    _deduplicate_papers,
    _title_similarity,
)


class TestTitleSimilarity:
    """Tests for title similarity calculation."""

    def test_identical_titles(self):
        """Test that identical titles have similarity of 1.0."""
        title = "AI Agents for Reasoning Tasks"
        assert _title_similarity(title, title) == 1.0

    def test_similar_titles(self):
        """Test that similar titles have high similarity."""
        title1 = "AI Agents for Reasoning Tasks"
        title2 = "AI Agents for Reasoning Tasks: A Study"
        similarity = _title_similarity(title1, title2)
        assert similarity > 0.7

    def test_different_titles(self):
        """Test that different titles have low similarity."""
        title1 = "AI Agents for Reasoning Tasks"
        title2 = "Weather Forecasting with Machine Learning"
        similarity = _title_similarity(title1, title2)
        assert similarity < 0.5

    def test_case_insensitive(self):
        """Test that similarity is case insensitive."""
        title1 = "AI Agents"
        title2 = "ai agents"
        assert _title_similarity(title1, title2) == 1.0

    def test_whitespace_handling(self):
        """Test that whitespace is normalized."""
        title1 = "AI Agents  "
        title2 = "  AI Agents"
        assert _title_similarity(title1, title2) == 1.0


class TestDeduplication:
    """Tests for paper deduplication."""

    def test_empty_list(self):
        """Test deduplication of empty list."""
        assert _deduplicate_papers([]) == []

    def test_no_duplicates(self, sample_papers):
        """Test that unique papers are all kept."""
        result = _deduplicate_papers(sample_papers)
        assert len(result) == len(sample_papers)

    def test_removes_exact_duplicates(self, sample_paper):
        """Test that exact duplicates are removed."""
        papers = [sample_paper, sample_paper.copy()]
        result = _deduplicate_papers(papers)
        assert len(result) == 1

    def test_removes_similar_titles(self):
        """Test that papers with similar titles are deduplicated."""
        papers = [
            {"title": "AI Agents for Complex Reasoning", "source": "arXiv"},
            {"title": "AI Agents for Complex Reasoning: A Study", "source": "HuggingFace"},
        ]
        result = _deduplicate_papers(papers, threshold=0.85)
        assert len(result) == 1

    def test_keeps_different_papers(self):
        """Test that papers with different titles are kept."""
        papers = [
            {"title": "AI Agents for Reasoning", "source": "arXiv"},
            {"title": "Weather Prediction Methods", "source": "HuggingFace"},
        ]
        result = _deduplicate_papers(papers)
        assert len(result) == 2

    def test_respects_threshold(self):
        """Test that similarity threshold is respected."""
        papers = [
            {"title": "AI Agents", "source": "arXiv"},
            {"title": "AI Agent", "source": "HuggingFace"},  # Very similar
        ]
        # With high threshold, should keep both
        result_high = _deduplicate_papers(papers, threshold=0.99)
        assert len(result_high) == 2

        # With low threshold, should deduplicate
        result_low = _deduplicate_papers(papers, threshold=0.7)
        assert len(result_low) == 1


class TestFetchAIResearch:
    """Tests for the main fetch_ai_research function."""

    def test_fetch_ai_research_aggregates_sources(self):
        """Test that research is aggregated from multiple sources."""
        with patch("src.research_fetcher.fetch_arxiv_papers") as mock_arxiv, \
             patch("src.research_fetcher.fetch_huggingface_papers") as mock_hf, \
             patch("src.research_fetcher.fetch_pwc_papers") as mock_pwc, \
             patch("src.research_fetcher.fetch_blog_posts") as mock_blogs:

            mock_arxiv.return_value = [{"title": "arXiv Paper", "published_at": "2024-01-15"}]
            mock_hf.return_value = [{"title": "HF Paper", "published_at": "2024-01-14"}]
            mock_pwc.return_value = [{"title": "PWC Paper", "published_at": "2024-01-13"}]
            mock_blogs.return_value = [{"title": "Blog Post", "published_at": "2024-01-12"}]

            result = fetch_ai_research(max_results=10)

            assert len(result) == 4
            # Should be sorted by date (most recent first)
            assert result[0]["title"] == "arXiv Paper"

    def test_fetch_ai_research_handles_source_failure(self):
        """Test that research is still returned if one source fails."""
        with patch("src.research_fetcher.fetch_arxiv_papers") as mock_arxiv, \
             patch("src.research_fetcher.fetch_huggingface_papers") as mock_hf, \
             patch("src.research_fetcher.fetch_pwc_papers") as mock_pwc, \
             patch("src.research_fetcher.fetch_blog_posts") as mock_blogs:

            mock_arxiv.return_value = [{"title": "arXiv Paper", "published_at": "2024-01-15"}]
            mock_hf.side_effect = Exception("API Error")
            mock_pwc.return_value = []
            mock_blogs.return_value = []

            result = fetch_ai_research(max_results=10)

            # Should still get arXiv results
            assert len(result) == 1
            assert result[0]["title"] == "arXiv Paper"

    def test_fetch_ai_research_deduplicates(self):
        """Test that duplicate papers are removed."""
        with patch("src.research_fetcher.fetch_arxiv_papers") as mock_arxiv, \
             patch("src.research_fetcher.fetch_huggingface_papers") as mock_hf, \
             patch("src.research_fetcher.fetch_pwc_papers") as mock_pwc, \
             patch("src.research_fetcher.fetch_blog_posts") as mock_blogs:

            # Same paper from two sources
            paper1 = {"title": "AI Agents Research Paper", "published_at": "2024-01-15", "source": "arXiv"}
            paper2 = {"title": "AI Agents Research Paper", "published_at": "2024-01-15", "source": "HF"}

            mock_arxiv.return_value = [paper1]
            mock_hf.return_value = [paper2]
            mock_pwc.return_value = []
            mock_blogs.return_value = []

            result = fetch_ai_research(max_results=10)

            # Should be deduplicated to 1
            assert len(result) == 1

    def test_fetch_ai_research_respects_max_results(self):
        """Test that max_results limit is respected."""
        with patch("src.research_fetcher.fetch_arxiv_papers") as mock_arxiv, \
             patch("src.research_fetcher.fetch_huggingface_papers") as mock_hf, \
             patch("src.research_fetcher.fetch_pwc_papers") as mock_pwc, \
             patch("src.research_fetcher.fetch_blog_posts") as mock_blogs:

            # Use distinct titles to avoid deduplication (short similar titles get deduplicated)
            distinct_titles = [
                "Quantum Computing Advances",
                "Neural Network Optimization",
                "Reinforcement Learning Breakthrough",
                "Natural Language Processing Study",
                "Computer Vision Research",
                "Robotics and Automation",
                "Machine Learning Theory",
                "Deep Learning Applications",
                "AI Safety Framework",
                "Transformer Architecture",
            ]
            papers = [{"title": distinct_titles[i], "published_at": f"2024-01-{15-i:02d}"} for i in range(10)]
            mock_arxiv.return_value = papers
            mock_hf.return_value = []
            mock_pwc.return_value = []
            mock_blogs.return_value = []

            result = fetch_ai_research(max_results=3)

            assert len(result) == 3

    def test_fetch_ai_research_sorts_by_date(self):
        """Test that results are sorted by date (most recent first)."""
        with patch("src.research_fetcher.fetch_arxiv_papers") as mock_arxiv, \
             patch("src.research_fetcher.fetch_huggingface_papers") as mock_hf, \
             patch("src.research_fetcher.fetch_pwc_papers") as mock_pwc, \
             patch("src.research_fetcher.fetch_blog_posts") as mock_blogs:

            mock_arxiv.return_value = [{"title": "Old Paper", "published_at": "2024-01-01"}]
            mock_hf.return_value = [{"title": "New Paper", "published_at": "2024-01-15"}]
            mock_pwc.return_value = []
            mock_blogs.return_value = []

            result = fetch_ai_research(max_results=10)

            assert result[0]["title"] == "New Paper"
            assert result[1]["title"] == "Old Paper"
