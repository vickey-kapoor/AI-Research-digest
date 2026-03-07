"""Centralized constants for AI Research WhatsApp Digest."""

import os

# Application info
APP_NAME = "AI Research WhatsApp Digest"
APP_VERSION = "1.0.0"
USER_AGENT = f"{APP_NAME}/{APP_VERSION} (https://github.com/vickey-kapoor/ai-research-whatsapp-digest)"

# Network settings
REQUEST_TIMEOUT = 30  # seconds
MAX_RETRIES = 3

# Deduplication settings
DEDUP_SIMILARITY_THRESHOLD = float(os.getenv("DEDUP_SIMILARITY_THRESHOLD", "0.85"))

# Digest settings
DIGEST_MAX_RESULTS = int(os.getenv("DIGEST_MAX_RESULTS", "10"))
WHATSAPP_MAX_CHARS = 1600

# Keywords for AI Agents & Reasoning research
AGENT_REASONING_KEYWORDS = [
    "AI agent",
    "autonomous agent",
    "reasoning",
    "chain of thought",
    "CoT",
    "ReAct",
    "tool use",
    "planning",
    "multi-agent",
    "agentic",
    "llm",
    "language model",
]

# Simplified keywords for filtering (lowercase)
FILTER_KEYWORDS = [
    "agent",
    "reasoning",
    "chain of thought",
    "cot",
    "react",
    "tool use",
    "tool-use",
    "planning",
    "multi-agent",
    "agentic",
    "autonomous",
    "llm",
    "language model",
    "gemini",
    "llama",
]

# arXiv categories
ARXIV_CATEGORIES = ["cs.AI", "cs.LG", "cs.CL", "cs.MA"]

# Blog RSS feeds
BLOG_FEEDS = {
    "Google AI": "https://blog.google/technology/ai/rss/",
    "DeepMind": "https://deepmind.google/blog/rss.xml",
    "Meta AI": "https://ai.meta.com/blog/rss/",
}

# API URLs
ARXIV_API_URL = "https://export.arxiv.org/api/query"
HF_DAILY_PAPERS_URL = "https://huggingface.co/api/daily_papers"
PWC_API_URL = "https://paperswithcode.com/api/v1/papers/"
