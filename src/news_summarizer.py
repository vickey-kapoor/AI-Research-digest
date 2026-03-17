"""Generate developer-focused summaries for AI product updates."""

from openai import OpenAI

from src.ai_text import sanitize_prompt_text
from src.constants import OPENAI_MODEL, OPENAI_TEMPERATURE, OPENAI_MAX_TOKENS_BUNDLE
from src.logger import get_logger
from src.utils.retry import retry_with_backoff

logger = get_logger(__name__)


def _prepare_inputs(research: dict) -> tuple[str, str, str]:
    """Sanitize and extract title, authors, and description from a research item."""
    title = sanitize_prompt_text(research.get("title", ""), 200)
    authors = sanitize_prompt_text(research.get("authors", "Unknown"), 100)
    description = sanitize_prompt_text(research.get("description", ""), 800)
    return title, authors, description


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


def summarize_research_bundle(research: dict, api_key: str) -> dict:
    """
    Generate both short and detailed summaries in a single model call.

    Returns the original paper unchanged if the request fails.
    """
    if not api_key:
        return research

    client = OpenAI(api_key=api_key)
    title, authors, abstract = _prepare_inputs(research)

    prompt = f"""You explain AI research to people with no technical background.

Paper: {title}
Authors: {authors}
Abstract: {abstract}

Write two outputs:

SHORT_SUMMARY:
- 4-5 simple sentences
- explain the problem, what they built, how it works in everyday terms, and why a regular person should care

DETAILED_SUMMARY:
- 8-12 paragraphs
- explain the big picture, what they did, why it is clever, real-world impact, and the bottom line

RULES:
- No jargon
- No technical terms like model, algorithm, neural network, training, parameters, architecture, benchmark, transformer, LLM
- Use everyday analogies
- Be warm and conversational

Respond in exactly this format:
SHORT_SUMMARY:
<text>

DETAILED_SUMMARY:
<text>"""

    try:
        content = _call_openai(client, prompt)
        parts = content.split("DETAILED_SUMMARY:", maxsplit=1)
        if len(parts) != 2 or "SHORT_SUMMARY:" not in parts[0]:
            return research

        short_summary = parts[0].split("SHORT_SUMMARY:", maxsplit=1)[1].strip()
        detailed_summary = parts[1].strip()
        if not short_summary and not detailed_summary:
            return research

        research_with_summaries = research.copy()
        if short_summary:
            research_with_summaries["summary"] = short_summary
        if detailed_summary:
            research_with_summaries["detailed_summary"] = detailed_summary
        return research_with_summaries
    except Exception:
        logger.warning("Could not generate summaries")
        return research
