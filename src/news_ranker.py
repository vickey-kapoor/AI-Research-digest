"""Rank AI lab developments by significance using OpenAI."""

import json

from openai import OpenAI

from src.ai_text import sanitize_prompt_text
from src.constants import OPENAI_MODEL, OPENAI_MAX_TOKENS_RANKING
from src.logger import get_logger
from src.topic_config import get_feedback_weights
from src.utils.retry import retry_with_backoff

logger = get_logger(__name__)
_sanitize_text = sanitize_prompt_text


@retry_with_backoff(exceptions=(Exception,))
def _call_openai_ranking(client: OpenAI, prompt: str):
    """Make an OpenAI API call for ranking with retry logic."""
    return client.chat.completions.create(
        model=OPENAI_MODEL,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=OPENAI_MAX_TOKENS_RANKING,
        temperature=0,
    )


def rank_research(research: list[dict], api_key: str) -> dict:
    """
    Use OpenAI to select the most significant AI lab development of the day.

    Args:
        research: List of candidate item dictionaries
        api_key: OpenAI API key

    Returns:
        The most significant item
    """
    if not research:
        raise ValueError("No research to rank")

    if len(research) == 1:
        return research[0]

    # Load feedback weights and reorder by preference before sending to LLM
    try:
        weights = get_feedback_weights()
        if weights:
            # Sort papers so preferred topics come first (higher weight = earlier)
            research = sorted(
                research,
                key=lambda r: weights.get(r.get("topic_id", ""), 1.0),
                reverse=True,
            )
    except Exception:
        pass  # Non-critical — proceed with original order

    client = OpenAI(api_key=api_key)

    # Prepare item summary for the prompt (with sanitization)
    research_text = "\n\n".join(
        f"[{i+1}] Title: {_sanitize_text(r.get('title', ''), 200)}\nSource: {_sanitize_text(r.get('source', ''), 50)}\nType: {r.get('type', 'announcement')}\nSummary: {_sanitize_text(r.get('summary', ''), 400)}"
        for i, r in enumerate(research)
    )

    prompt = f"""You are tracking what the frontier AI labs ship — OpenAI, Anthropic, Google DeepMind, Meta, Mistral, xAI, Alibaba/Qwen, DeepSeek, NVIDIA and their peers.
Your job is to pick the single most significant AI lab development from today's list.

Rank by how much the item changes what is actually available or known today.
Prioritize:
1. New frontier model releases — a lab shipping a model, version, or checkpoint, with capability or pricing detail
2. Product and API launches that change what developers can build — new endpoints, modalities, context limits, availability, pricing moves
3. Lab research with concrete results — technical reports, scaling findings, training or post-training methods, with numbers
4. Agentic capability milestones — tool use, computer use, long-horizon or coding agents shipped or measured
5. Benchmark results that move the frontier — SWE-bench, ARC-AGI, GPQA, AIME and similar, especially with verified methodology
6. System cards, safety frameworks and evaluation reports accompanying a frontier release
7. Open-weight releases that shift what is freely runnable
8. Compute and infrastructure news that changes training or serving economics at frontier scale

Deprioritize:
- Third-party commentary, opinion, or speculation about a lab rather than an announcement from one
- Incremental SDK or library point releases with no capability change
- Reposts, roundups, and coverage of an announcement already made days ago
- Hiring, partnerships, funding rounds, leadership changes, legal disputes
- Marketing posts with no model, number, or shipping date

Prefer first-party lab announcements over coverage of them. When two items describe the same launch, pick the one closest to the source.

Items:
{research_text}

Return the single most important item as JSON: {{"index": N, "reason": "one sentence on why this is the most significant lab development today"}}"""

    try:
        response = _call_openai_ranking(client, prompt)

        content = response.choices[0].message.content
        if content:
            # Try JSON parse first, fall back to plain number
            try:
                result = json.loads(content.strip())
                selected_index = int(result["index"]) - 1
            except (json.JSONDecodeError, KeyError):
                selected_index = int(content.strip()) - 1
            if 0 <= selected_index < len(research):
                return research[selected_index]
    except (ValueError, IndexError, TypeError, AttributeError):
        pass
    except Exception:
        logger.error("Failed to rank items with AI")

    # Fallback to first paper if parsing fails
    return research[0]
