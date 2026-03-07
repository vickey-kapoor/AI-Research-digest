"""Shared pytest fixtures for AI Research WhatsApp Digest tests."""

import pytest
from unittest.mock import Mock, patch


@pytest.fixture
def sample_paper():
    """Sample research paper dictionary."""
    return {
        "title": "Autonomous AI Agents with Chain-of-Thought Reasoning",
        "description": "We present a novel approach to autonomous AI agents using chain-of-thought reasoning for complex task planning.",
        "source": "arXiv",
        "url": "https://arxiv.org/abs/2401.12345",
        "published_at": "2024-01-15T00:00:00",
        "type": "research",
        "authors": "Alice Smith, Bob Jones",
        "topics": ["cs.AI", "cs.LG"],
    }


@pytest.fixture
def sample_papers(sample_paper):
    """List of sample research papers."""
    return [
        sample_paper,
        {
            "title": "Multi-Agent Systems for Collaborative Problem Solving",
            "description": "A study on multi-agent collaboration in complex environments.",
            "source": "Hugging Face",
            "url": "https://huggingface.co/papers/2401.54321",
            "published_at": "2024-01-14T00:00:00",
            "type": "research",
            "authors": "Carol White, David Brown",
            "topics": ["AI Agents", "Reasoning"],
        },
        {
            "title": "ReAct: Reasoning and Acting in Language Models",
            "description": "Combining reasoning and acting capabilities in language models.",
            "source": "Papers With Code",
            "url": "https://paperswithcode.com/paper/react",
            "published_at": "2024-01-13T00:00:00",
            "type": "research",
            "authors": "Eve Green",
            "topics": ["AI Agents", "Reasoning"],
        },
    ]


@pytest.fixture
def sample_paper_with_summary(sample_paper):
    """Sample paper with generated summary."""
    paper = sample_paper.copy()
    paper["summary"] = "Scientists created a smarter AI that can plan tasks step by step, like how you might plan your day. This could make future AI assistants much better at helping with complex tasks."
    return paper


@pytest.fixture
def sample_paper_with_detailed_summary(sample_paper_with_summary):
    """Sample paper with detailed summary for PDF."""
    paper = sample_paper_with_summary.copy()
    paper["detailed_summary"] = """**The Big Picture**

Imagine you have a really clever assistant who not only follows your instructions but actually thinks through problems step by step, just like you would. That's what these researchers are trying to create.

**What They Did**

The team built a new kind of AI that plans its actions before taking them. Think of it like a chess player who thinks several moves ahead instead of just reacting to what's in front of them.

**Why It's Clever**

Most AIs just respond quickly without much thought. This one actually reasons through problems, which makes it much better at handling unexpected situations.

**Real World Impact**

In a few years, your phone assistant might be able to plan your whole day, book appointments, and handle complex tasks without you having to give step-by-step instructions.

**The Bottom Line**

Scientists taught AI to think before it acts, making it much smarter at handling real-world tasks."""
    return paper


@pytest.fixture
def mock_openai_response():
    """Mock OpenAI API response."""
    mock_response = Mock()
    mock_response.choices = [Mock()]
    mock_response.choices[0].message.content = "1"
    return mock_response


@pytest.fixture
def mock_openai_summary_response():
    """Mock OpenAI API response for summarization."""
    mock_response = Mock()
    mock_response.choices = [Mock()]
    mock_response.choices[0].message.content = "Scientists created a smarter AI that can plan tasks step by step, like how you might plan your day."
    return mock_response


@pytest.fixture
def mock_arxiv_feed():
    """Mock arXiv feed response."""
    return {
        "bozo": False,
        "entries": [
            {
                "title": "Test Paper on AI Agents",
                "summary": "This paper presents a novel approach to AI agents.",
                "link": "https://arxiv.org/abs/2401.00001",
                "published": "2024-01-15T00:00:00Z",
                "authors": [{"name": "Test Author"}],
                "tags": [{"term": "cs.AI"}, {"term": "cs.LG"}],
            }
        ],
    }


@pytest.fixture
def mock_huggingface_response():
    """Mock Hugging Face API response."""
    return [
        {
            "paper": {
                "id": "2401.00001",
                "title": "Agent-Based AI Research",
                "summary": "A paper about AI agents and reasoning.",
                "publishedAt": "2024-01-15T00:00:00Z",
                "authors": [{"name": "HF Author"}],
            }
        }
    ]


@pytest.fixture
def mock_pwc_response():
    """Mock Papers With Code API response."""
    return {
        "results": [
            {
                "id": "test-paper",
                "title": "Planning in Multi-Agent Systems",
                "abstract": "Research on planning and reasoning in multi-agent environments.",
                "published": "2024-01-15",
                "authors": ["PWC Author"],
                "url_abs": "https://paperswithcode.com/paper/test-paper",
            }
        ]
    }


@pytest.fixture
def mock_blog_feed():
    """Mock blog RSS feed response."""
    return {
        "bozo": False,
        "entries": [
            {
                "title": "New Developments in AI Agents",
                "summary": "Google AI announces new agent capabilities.",
                "link": "https://blog.google/technology/ai/test-post",
                "published": "2024-01-15T00:00:00Z",
                "published_parsed": (2024, 1, 15, 0, 0, 0, 0, 15, 0),
            }
        ],
    }


@pytest.fixture
def env_vars(monkeypatch):
    """Set up required environment variables for testing."""
    monkeypatch.setenv("TWILIO_ACCOUNT_SID", "test_sid")
    monkeypatch.setenv("TWILIO_AUTH_TOKEN", "test_token")
    monkeypatch.setenv("TWILIO_WHATSAPP_NUMBER", "whatsapp:+14155238886")
    monkeypatch.setenv("YOUR_WHATSAPP_NUMBER", "whatsapp:+14081234567")
    monkeypatch.setenv("OPENAI_API_KEY", "test_openai_key")
