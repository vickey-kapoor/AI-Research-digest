"""Generate structured lab-release briefs for AI lab developments."""

import json

from openai import OpenAI

from src.ai_text import sanitize_prompt_text
from src.constants import OPENAI_MODEL, OPENAI_TEMPERATURE, OPENAI_MAX_TOKENS_BUNDLE
from src.logger import get_logger
from src.utils.retry import retry_with_backoff

logger = get_logger(__name__)


def _prepare_inputs(research: dict) -> tuple[str, str, str]:
    """Sanitize and extract title, source, and summary from an item."""
    title = sanitize_prompt_text(research.get("title", ""), 200)
    source = sanitize_prompt_text(research.get("source", "Unknown"), 100)
    summary = sanitize_prompt_text(research.get("summary", ""), 800)
    return title, source, summary


@retry_with_backoff(max_retries=2, base_delay=1.0, exceptions=(Exception,))
def _call_openai(client: OpenAI, prompt: str) -> str:
    """Make an OpenAI API call with retry logic."""
    response = client.chat.completions.create(
        model=OPENAI_MODEL,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=OPENAI_MAX_TOKENS_BUNDLE,
        temperature=OPENAI_TEMPERATURE,
    )
    return response.choices[0].message.content.strip()


# Structured fields produced by the model, in the order they read best.
SUMMARY_FIELDS = (
    "what_shipped",
    "capabilities",
    "availability",
    "why_it_matters",
    "caveats",
    "release_type",
)

# Fields rendered into the long-form PDF body, with their display labels.
DETAIL_SECTIONS = (
    ("what_shipped", "What shipped"),
    ("capabilities", "Capabilities"),
    ("availability", "Availability"),
    ("why_it_matters", "Why it matters"),
    ("caveats", "Caveats"),
)


def summarize_research_bundle(research: dict, api_key: str) -> dict:
    """Generate a structured lab-release brief in a single model call.

    Returns the original item unchanged if the request fails.
    """
    if not api_key:
        return research

    client = OpenAI(api_key=api_key)
    title, source, description = _prepare_inputs(research)

    prompt = f"""You are briefing an engineer who tracks what the frontier AI labs ship.
Be concrete and factual. No hype, no marketing language. Prefer specifics — model names, numbers, prices, dates — over adjectives.
If the description does not state something, say so rather than inventing it.

Item title: {title}
Source: {source}
Description: {description}

Return JSON (no markdown fences):
{{
  "what_shipped": "One sentence — which lab shipped what: the model, product, API, paper, or result, named precisely",
  "capabilities": "2-3 sentences — what it can do that matters: modalities, context length, benchmark numbers, speed or quality claims, with the figures given",
  "availability": "1-2 sentences — how to get it: API, app, open weights, waitlist, preview or GA, regions, pricing and tiers if stated. Say 'not stated' if the description does not say",
  "why_it_matters": "1-2 sentences — what this changes for people building on these models: new capability unlocked, cost shift, or competitive move",
  "caveats": "1-2 sentences — the honest limits: unverified claims, benchmark caveats, limited access, missing detail, or where the announcement outruns the evidence",
  "release_type": "model, product, api, research, open-weights, infrastructure, safety, or policy — tag what kind of development this is"
}}"""

    try:
        content = _call_openai(client, prompt)
        # Strip markdown fences if model adds them
        content = content.strip()
        if content.startswith("```"):
            content = content.split("\n", 1)[-1]
        if content.endswith("```"):
            content = content.rsplit("```", 1)[0]
        content = content.strip()

        parsed = json.loads(content)
        if not isinstance(parsed, dict):
            return research

        research_with_summaries = research.copy()

        # Store structured fields
        for key in SUMMARY_FIELDS:
            if parsed.get(key):
                research_with_summaries[key] = parsed[key]

        # Build short summary for backward compat (Telegram fallback, KV, etc.)
        parts = []
        if parsed.get("what_shipped"):
            parts.append(parsed["what_shipped"])
        if parsed.get("capabilities"):
            parts.append(parsed["capabilities"])
        if parts:
            research_with_summaries["summary"] = " ".join(parts)

        # Build detailed summary for PDF
        detail_parts = []
        for key, label in DETAIL_SECTIONS:
            if parsed.get(key):
                detail_parts.append(f"**{label}**\n{parsed[key]}")
        if detail_parts:
            research_with_summaries["detailed_summary"] = "\n\n".join(detail_parts)

        return research_with_summaries
    except Exception:
        logger.warning("Could not generate summaries")
        return research
