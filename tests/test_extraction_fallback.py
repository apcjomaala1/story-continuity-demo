import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.constants import ProviderConfig
from src.extraction import attach_fact_line_refs, heuristic_fact_limit


def make_provider() -> ProviderConfig:
    return ProviderConfig(
        mode="Ollama API",
        extraction_depth="Normal scene/chapter excerpt",
        ollama_url="http://localhost:11434",
        ollama_model="llama3.2:3b",
        gemini_url="https://generativelanguage.googleapis.com/v1beta",
        gemini_key="",
        gemini_model="gemini-2.5-flash",
        claude_url="https://api.anthropic.com/v1",
        claude_key="",
        claude_model="claude-3-5-sonnet-latest",
        api_url="https://api.openai.com/v1",
        api_key="",
        api_model="gpt-4o-mini",
        fallback_to_heuristic=True,
    )


def test_heuristic_fact_limit_scales_for_oversized_text() -> None:
    provider = make_provider()

    assert heuristic_fact_limit("x" * 15000, provider) > heuristic_fact_limit("x" * 1500, provider)


def test_attach_fact_line_refs_uses_evidence_location() -> None:
    text = "First line.\nLiora Vale owns a trunk.\nThird line."
    facts = [{"type": "possession", "subject": "Liora Vale", "predicate": "owns", "object": "trunk", "evidence": "Liora Vale owns a trunk."}]

    [fact] = attach_fact_line_refs(text, facts)

    assert fact["line_start"] == 2
    assert fact["line_end"] == 2
