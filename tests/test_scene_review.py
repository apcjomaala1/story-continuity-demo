import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import ui
from src.constants import ProviderConfig


def make_provider() -> ProviderConfig:
    return ProviderConfig(
        mode="Ollama local LLM",
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


def test_scene_review_extracts_and_checks_against_memory(monkeypatch) -> None:
    provider = make_provider()
    metadata = {"source": "draft scene", "chapter": 1, "story_order": 1}
    facts = [{"id": "fact-1", "type": "event"}]
    memory = [{"id": "memory-1", "type": "event"}]
    issues = [object()]
    calls = {}

    def fake_extract(text, metadata_arg, provider_arg, *, split_over_limit):
        calls["extract"] = (text, metadata_arg, provider_arg, split_over_limit)
        return facts, "notice", "debug"

    def fake_check(text, facts_arg, memory_arg):
        calls["check"] = (text, facts_arg, memory_arg)
        return issues

    monkeypatch.setattr(ui, "extract_facts_for_text", fake_extract)
    monkeypatch.setattr(ui, "check_continuity", fake_check)

    result = ui.review_text_against_memory(
        "Mara is Dain's boss.",
        metadata,
        provider,
        memory,
        split_over_limit=True,
    )

    assert result == (facts, issues, "notice", "debug")
    assert calls["extract"] == ("Mara is Dain's boss.", metadata, provider, True)
    assert calls["check"] == ("Mara is Dain's boss.", facts, memory)


def test_scene_review_skips_check_when_memory_is_empty(monkeypatch) -> None:
    provider = make_provider()
    metadata = {"source": "draft scene", "chapter": 1, "story_order": 1}
    facts = [{"id": "fact-1", "type": "event"}]

    def fake_extract(text, metadata_arg, provider_arg, *, split_over_limit):
        return facts, "notice", ""

    def fail_check(*args, **kwargs):
        raise AssertionError("empty memory should not be checked")

    monkeypatch.setattr(ui, "extract_facts_for_text", fake_extract)
    monkeypatch.setattr(ui, "check_continuity", fail_check)

    result = ui.review_text_against_memory(
        "Mara arrived.",
        metadata,
        provider,
        [],
        split_over_limit=False,
    )

    assert result == (facts, [], "notice", "")
