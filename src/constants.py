"""Centralized constants for AI Dev Digest."""

import os

# Application info
APP_NAME = "AI Dev Digest"
APP_VERSION = "2.0.0"
USER_AGENT = f"{APP_NAME}/{APP_VERSION} (https://github.com/vickey-kapoor/ai-research-digest)"

# Network settings
REQUEST_TIMEOUT = 30  # seconds
MAX_RETRIES = 3

# Deduplication settings
DEDUP_SIMILARITY_THRESHOLD = float(os.getenv("DEDUP_SIMILARITY_THRESHOLD", "0.85"))

# Digest settings
DIGEST_MAX_RESULTS = int(os.getenv("DIGEST_MAX_RESULTS", "10"))

# OpenAI model settings
OPENAI_MODEL = "gpt-4o-mini"
OPENAI_TEMPERATURE = 0.7
OPENAI_MAX_TOKENS_RANKING = 150
OPENAI_MAX_TOKENS_SUMMARY = 500
OPENAI_MAX_TOKENS_DETAILED = 1500
OPENAI_MAX_TOKENS_BUNDLE = 1800

# Telegram settings
TELEGRAM_API_URL = "https://api.telegram.org/bot{token}/sendMessage"
TELEGRAM_MAX_MESSAGE_LENGTH = 4096

# Data cap settings
PAPERS_CAP = 500
DIGEST_CAP_DAYS = 90

# Thread pool settings
THREAD_POOL_WORKERS = 2

# Persona: engineer/researcher tracking what the frontier AI labs ship.
# Default keyword set — broad lab-development vocabulary used as a fallback when
# no topic-config keywords are passed through. Topic-level filtering (in
# topic_config.DEFAULT_TOPICS) is the primary signal.
#
# Keywords are matched as case-insensitive substrings against title + summary,
# so avoid short tokens that hide inside common words (e.g. "api" matches
# "rapid"). Prefer multi-word identifiers.
FILTER_KEYWORDS = [
    # Model releases
    "model release", "new model", "frontier model", "flagship model",
    "reasoning model", "model family", "we're releasing", "introducing claude",
    "introducing gpt", "gemini", "claude", "llama", "mistral", "qwen",
    "deepseek", "grok", "gemma", "phi-", "command r", "nova",
    # Product / platform
    "developer api", "api access", "api pricing", "batch api",
    "fine-tuning api", "responses api", "assistants api", "public beta",
    "general availability", "context window", "developer platform",
    "now available", "rolling out",
    # Research
    "technical report", "research paper", "scaling law", "pretraining",
    "post-training", "reinforcement learning", "distillation",
    "mixture of experts", "chain of thought", "test-time compute",
    "long context", "rlhf",
    # Agents / tooling
    "agentic", "agent", "tool use", "function calling", "computer use",
    "model context protocol", "coding agent", "multi-agent",
    # Benchmarks
    "benchmark", "eval", "swe-bench", "gpqa", "arc-agi", "aime",
    "state of the art", "leaderboard", "frontiermath",
    # Safety / system cards
    "system card", "model card", "red-teaming", "red teaming", "jailbreak",
    "responsible scaling", "preparedness framework", "frontier safety",
    "interpretability", "alignment",
    # Open weights
    "open weights", "open-weight", "open source model", "open model",
    # Infrastructure
    "inference", "quantization", "training run", "tpu", "blackwell",
    "trainium", "gpu cluster", "serving",
    # Multimodal
    "multimodal", "vision-language", "image generation", "video generation",
    "text-to-video", "speech model", "world model",
]

# Keywords to exclude non-technical corporate news (lowercase)
EXCLUDE_KEYWORDS = [
    "hiring",
    "careers",
    "joins",
    "leadership",
    "appointed",
    "raises",
    "funding round",
    "series a",
    "series b",
    "ipo",
    "lawsuit",
    "trademark",
]

# Blog RSS feeds — frontier AI labs and the platforms they ship on.
# Every URL here was verified to return a parseable feed. Anthropic publishes
# no public RSS feed for its news/research/engineering posts, so Anthropic
# coverage comes through Hacker News keywords and Hugging Face papers instead.
BLOG_FEEDS = {
    "OpenAI": "https://openai.com/news/rss.xml",
    "Google DeepMind": "https://deepmind.google/blog/rss.xml",
    "Google AI": "https://blog.google/technology/ai/rss/",
    "Google Research": "https://research.google/blog/rss/",
    "Meta AI": "https://engineering.fb.com/category/ml-applications/feed/",
    "Mistral AI": "https://mistral.ai/rss.xml",
    "Qwen": "https://qwenlm.github.io/blog/index.xml",
    "Hugging Face": "https://huggingface.co/blog/feed.xml",
    "NVIDIA": "https://blogs.nvidia.com/feed/",
    "Together AI": "https://www.together.ai/blog/rss.xml",
    "EleutherAI": "https://blog.eleuther.ai/index.xml",
    "AWS Machine Learning": "https://aws.amazon.com/blogs/machine-learning/feed/",
}

# Minimum candidate posts pulled per blog feed before keyword filtering.
# Without a floor, adding feeds shrinks each feed's share to a single post,
# so a lab that posted a few consumer items ahead of its release loses it.
# The feed is fetched in full either way, so this costs no extra requests.
BLOG_MIN_PER_SOURCE = 5

# GitHub repos whose releases mark a shipped lab development — official model
# SDKs plus the serving/runtime stacks new models land in first.
GITHUB_REPOS: list[str] = [
    "openai/openai-python",
    "anthropics/anthropic-sdk-python",
    "googleapis/python-genai",
    "huggingface/transformers",
    "vllm-project/vllm",
    "ggml-org/llama.cpp",
    "modelcontextprotocol/servers",
]

# Hacker News filter keywords — frontier lab launches and announcements.
# This is the primary channel for Anthropic news, which has no RSS feed.
HN_KEYWORDS = [
    # Labs by name
    "OpenAI", "Anthropic", "Claude", "GPT-", "DeepMind", "Gemini",
    "Mistral", "Qwen", "DeepSeek", "Llama", "Grok", "xAI", "Gemma",
    # Release language
    "model release", "new model", "frontier model", "reasoning model",
    "open weights", "open-weight", "system card", "technical report",
    # Capability surfaces
    "agentic", "computer use", "function calling", "model context protocol",
    "long context", "multimodal", "benchmark", "SWE-bench", "ARC-AGI",
    # Infra
    "inference", "TPU", "Blackwell", "training run",
]
HN_MIN_SCORE = 100
HN_MAX_STORIES = 5

# Hugging Face Daily Papers — lab research lands here first. Kept moderately
# permissive so lab technical reports surface the day they drop.
HF_MIN_UPVOTES = 20
HF_MAX_PAPERS = 5
