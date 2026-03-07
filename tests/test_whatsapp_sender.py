"""Unit tests for WhatsApp sender module."""

import pytest
from unittest.mock import Mock, patch

from src.whatsapp_sender import (
    _validate_url,
    _truncate,
    format_research_message,
    send_whatsapp_message,
    WHATSAPP_MAX_CHARS,
)


class TestUrlValidation:
    """Tests for URL validation."""

    def test_valid_https_url(self):
        """Test that valid HTTPS URLs are accepted."""
        url = "https://arxiv.org/abs/2401.12345"
        assert _validate_url(url) == url

    def test_valid_http_url(self):
        """Test that valid HTTP URLs are accepted."""
        url = "http://example.com/paper"
        assert _validate_url(url) == url

    def test_empty_url(self):
        """Test that empty URLs return empty string."""
        assert _validate_url("") == ""
        assert _validate_url(None) == ""

    def test_invalid_scheme(self):
        """Test that non-http(s) schemes are rejected."""
        assert _validate_url("ftp://example.com") == ""
        assert _validate_url("file:///etc/passwd") == ""

    def test_javascript_injection(self):
        """Test that JavaScript URLs are rejected."""
        assert _validate_url("javascript:alert(1)") == ""

    def test_data_url(self):
        """Test that data URLs are rejected."""
        assert _validate_url("data:text/html,<script>alert(1)</script>") == ""

    def test_xss_patterns(self):
        """Test that XSS patterns in URLs are rejected."""
        assert _validate_url("https://example.com/<script>alert(1)</script>") == ""
        assert _validate_url("https://example.com/onclick=alert(1)") == ""

    def test_whitespace_handling(self):
        """Test that whitespace is trimmed from URLs."""
        url = "  https://arxiv.org/abs/2401.12345  "
        assert _validate_url(url) == url.strip()

    def test_missing_netloc(self):
        """Test that URLs without host are rejected."""
        assert _validate_url("https:///path") == ""


class TestTruncate:
    """Tests for text truncation."""

    def test_short_text(self):
        """Test that short text is not truncated."""
        text = "Short text"
        assert _truncate(text, 100) == text

    def test_exact_length(self):
        """Test text at exactly max length."""
        text = "a" * 50
        assert _truncate(text, 50) == text

    def test_truncation(self):
        """Test that long text is truncated."""
        text = "This is a long sentence that needs truncation"
        result = _truncate(text, 20)
        assert len(result) <= 20
        assert result.endswith("...")

    def test_word_boundary(self):
        """Test that truncation respects word boundaries."""
        text = "This is a test sentence"
        result = _truncate(text, 15)
        assert result == "This is a..."


class TestFormatResearchMessage:
    """Tests for research message formatting."""

    def test_basic_formatting(self, sample_paper_with_summary):
        """Test basic message formatting."""
        message = format_research_message(sample_paper_with_summary)

        assert "*Daily AI Research*" in message
        assert sample_paper_with_summary["title"] in message
        assert sample_paper_with_summary["summary"] in message
        assert sample_paper_with_summary["source"] in message

    def test_empty_research(self):
        """Test formatting with empty research."""
        message = format_research_message({})
        assert "No research found today" in message

    def test_none_research(self):
        """Test formatting with None research."""
        message = format_research_message(None)
        assert "No research found today" in message

    def test_message_length_limit(self, sample_paper_with_summary):
        """Test that message respects WhatsApp character limit."""
        # Create paper with very long summary
        paper = sample_paper_with_summary.copy()
        paper["summary"] = "A" * 2000  # Very long summary

        message = format_research_message(paper)

        assert len(message) <= WHATSAPP_MAX_CHARS

    def test_url_validation_in_message(self, sample_paper_with_summary):
        """Test that URL is validated in message."""
        paper = sample_paper_with_summary.copy()
        paper["url"] = "javascript:alert(1)"

        message = format_research_message(paper)

        # Malicious URL should not appear in message
        assert "javascript:" not in message

    def test_author_truncation(self, sample_paper):
        """Test that long author lists are truncated."""
        paper = sample_paper.copy()
        paper["authors"] = "A" * 100
        paper["summary"] = "Test summary"

        message = format_research_message(paper)

        # Should contain truncated authors
        assert "..." in message or len(paper["authors"]) <= 60


class TestSendWhatsAppMessage:
    """Tests for WhatsApp message sending."""

    def test_send_message_success(self):
        """Test successful message sending."""
        with patch("src.whatsapp_sender.Client") as mock_client:
            mock_message = Mock()
            mock_message.sid = "SM123456"
            mock_client.return_value.messages.create.return_value = mock_message

            result = send_whatsapp_message(
                "test_sid",
                "test_token",
                "whatsapp:+14155238886",
                "whatsapp:+14081234567",
                "Test message",
            )

            assert result == "SM123456"
            mock_client.return_value.messages.create.assert_called_once_with(
                body="Test message",
                from_="whatsapp:+14155238886",
                to="whatsapp:+14081234567",
            )

    def test_send_message_failure(self):
        """Test message sending failure handling."""
        with patch("src.whatsapp_sender.Client") as mock_client:
            mock_client.return_value.messages.create.side_effect = Exception("API Error")

            with pytest.raises(Exception):
                send_whatsapp_message(
                    "test_sid",
                    "test_token",
                    "whatsapp:+14155238886",
                    "whatsapp:+14081234567",
                    "Test message",
                )
