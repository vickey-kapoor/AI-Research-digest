"""Generate ELI5 (Explain Like I'm 5) summaries for research papers."""

import re

from openai import OpenAI

from src.logger import get_logger

logger = get_logger(__name__)


def _sanitize_text(text: str, max_length: int = 500) -> str:
    """
    Sanitize text to prevent prompt injection.

    - Removes potential injection patterns
    - Limits length
    - Strips control characters
    """
    if not text:
        return ""

    # Remove control characters except newlines and tabs
    text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', '', text)

    # Remove potential prompt injection patterns
    injection_patterns = [
        r'ignore\s+(previous|above|all)\s+instructions?',
        r'disregard\s+(previous|above|all)',
        r'forget\s+(everything|previous|above)',
        r'new\s+instructions?:',
        r'system\s*:',
        r'assistant\s*:',
        r'user\s*:',
        r'\[INST\]',
        r'\[/INST\]',
        r'<\|.*?\|>',
    ]
    for pattern in injection_patterns:
        text = re.sub(pattern, '[FILTERED]', text, flags=re.IGNORECASE)

    # Truncate to max length
    if len(text) > max_length:
        text = text[:max_length] + "..."

    return text.strip()


def summarize_research(research: dict, api_key: str) -> dict:
    """
    Generate a simple, jargon-free summary for a research paper.

    Summaries are written so a kid or grandma could understand them.

    Args:
        research: Research paper dictionary with title, description, source, url, authors
        api_key: OpenAI API key

    Returns:
        Research dictionary with added 'summary' field
    """
    if not api_key:
        return research

    client = OpenAI(api_key=api_key)

    # Sanitize inputs to prevent prompt injection
    title = _sanitize_text(research.get("title", ""), 200)
    authors = _sanitize_text(research.get("authors", "Unknown"), 100)
    abstract = _sanitize_text(research.get("description", ""), 500)

    prompt = f"""You explain AI research to someone with NO technical background - like a grandma or a kid.

Paper: {title}
Authors: {authors}
Abstract: {abstract}

Write 4-5 simple sentences explaining:
1. What problem were the researchers trying to solve?
2. What did they build or discover? (use simple analogies)
3. How does it work in simple terms?
4. Why should a regular person care? How might this affect their life someday?

RULES:
- NO jargon (no "transformer", "architecture", "benchmark", "SOTA", "LLM", "neural network", "model", "parameters")
- Use everyday analogies (like teaching, cooking, organizing, finding things, etc.)
- Write like you're explaining to a curious grandma over coffee
- Be warm and conversational

Example good summary:
"You know how it's hard to find exactly what you're looking for when you have a messy desk? Researchers figured out a way to help AI organize information better so it can find the right answer faster. They taught it to sort through piles of information like a librarian who knows exactly where every book is. This could make future AI assistants much quicker at answering your questions - no more waiting around!"

Now write the summary:"""

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=500,
            temperature=0.7,
        )

        summary = response.choices[0].message.content.strip()
        research_with_summary = research.copy()
        research_with_summary["summary"] = summary
        return research_with_summary

    except Exception:
        logger.warning("Could not generate summary")
        return research


def summarize_research_detailed(research: dict, api_key: str) -> dict:
    """
    Generate a detailed, comprehensive summary for PDF reports.

    Much longer than WhatsApp summary - full grandma-friendly explanation.

    Args:
        research: Research paper dictionary with title, description, source, url, authors
        api_key: OpenAI API key

    Returns:
        Research dictionary with added 'detailed_summary' field
    """
    if not api_key:
        return research

    client = OpenAI(api_key=api_key)

    # Sanitize inputs
    title = _sanitize_text(research.get("title", ""), 200)
    authors = _sanitize_text(research.get("authors", "Unknown"), 100)
    abstract = _sanitize_text(research.get("description", ""), 800)

    prompt = f"""You are a patient, warm science communicator explaining cutting-edge AI research to someone with NO technical background - like explaining to your curious grandma who wants to understand what her grandkid is working on.

Paper: {title}
Authors: {authors}
Abstract: {abstract}

Write a DETAILED explanation (8-12 paragraphs) covering:

1. **The Big Picture** (2-3 paragraphs)
   - What area of AI is this about? Explain it simply.
   - Why are scientists working on this problem?
   - What was missing or broken before this research?

2. **What They Did** (2-3 paragraphs)
   - What did the researchers actually build or discover?
   - Use a vivid everyday analogy (cooking, gardening, teaching kids, organizing a closet, etc.)
   - Walk through how it works step by step, like explaining a recipe

3. **Why It's Clever** (1-2 paragraphs)
   - What makes this approach special or different?
   - What's the "aha!" moment?

4. **Real World Impact** (2-3 paragraphs)
   - How might this affect regular people's lives in 5-10 years?
   - Give concrete examples (your phone, your car, your doctor, shopping, etc.)
   - What problems could this help solve?

5. **The Bottom Line** (1 paragraph)
   - Summarize in 2-3 sentences what grandma should remember

RULES:
- NO jargon whatsoever
- NO technical terms (no "model", "algorithm", "neural network", "training", "parameters", "architecture", "benchmark")
- Use analogies from everyday life
- Write like you're having coffee with grandma
- Be warm, patient, and encouraging
- Use "you" and "your" to make it personal
- It's OK to be longer - grandma has time and wants to understand!

Now write the detailed explanation:"""

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=1500,
            temperature=0.7,
        )

        detailed_summary = response.choices[0].message.content.strip()
        research_with_summary = research.copy()
        research_with_summary["detailed_summary"] = detailed_summary
        return research_with_summary

    except Exception:
        logger.warning("Could not generate detailed summary")
        return research
