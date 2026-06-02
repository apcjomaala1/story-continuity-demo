import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import extraction
from src.continuity import check_continuity
from src.constants import ProviderConfig
from src.extraction import (
    attach_fact_line_refs,
    backfill_uncovered_relevant_facts,
    heuristic_extract,
    heuristic_fact_limit,
    recover_uncovered_relevant_facts,
)


def make_provider(*, fallback_to_heuristic: bool = True, retry_missed_with_ai: bool = False) -> ProviderConfig:
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
        fallback_to_heuristic=fallback_to_heuristic,
        retry_missed_with_ai=retry_missed_with_ai,
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


def test_coverage_backfill_adds_facts_for_missed_relevant_lines() -> None:
    text = "Liora Vale owns an iron key.\nCaelan Thorne is dead."
    facts = attach_fact_line_refs(
        text,
        [
            {
                "type": "possession",
                "subject": "Liora Vale",
                "predicate": "owns",
                "object": "iron key",
                "value": "",
                "evidence": "Liora Vale owns an iron key.",
            }
        ],
    )

    merged, notice = backfill_uncovered_relevant_facts(text, facts, {"source": "test", "chapter": 1, "story_order": 1})

    assert "backfilled" in notice
    assert any(
        fact["subject"] == "Caelan Thorne"
        and fact["type"] == "status"
        and fact["value"] == "dead"
        and fact["line_start"] == 2
        and fact["extraction"] == "heuristic_backfill"
        for fact in merged
    )


def test_coverage_backfill_does_not_duplicate_covered_lines() -> None:
    text = "Caelan Thorne is dead."
    facts = attach_fact_line_refs(
        text,
        [
            {
                "type": "status",
                "subject": "Caelan Thorne",
                "predicate": "status",
                "object": "",
                "value": "dead",
                "evidence": "Caelan Thorne is dead.",
            }
        ],
    )

    merged, notice = backfill_uncovered_relevant_facts(text, facts, {"source": "test", "chapter": 1, "story_order": 1})

    assert notice == ""
    assert merged == facts


def test_coverage_backfill_adds_note_when_relevant_line_cannot_be_parsed() -> None:
    text = "Liora Vale was unmistakably moonlit."

    merged, notice = backfill_uncovered_relevant_facts(text, [], {"source": "test", "chapter": 1, "story_order": 1})

    assert "backfilled" in notice
    assert any(
        fact["type"] == "event"
        and fact["subject"] == "Liora Vale"
        and fact["predicate"] == "mentions"
        and fact["line_start"] == 1
        and fact["extraction"] == "heuristic_backfill"
        and fact["confidence"] == 0.3
        for fact in merged
    )


def test_coverage_backfill_respects_disabled_heuristic_fallback() -> None:
    text = "Caelan Thorne is dead."

    merged, notice = backfill_uncovered_relevant_facts(
        text,
        [],
        {"source": "test", "chapter": 1, "story_order": 1},
        allow_heuristic_backfill=False,
    )

    assert merged == []
    assert "private backup reader is disabled" in notice


def test_recover_uncovered_relevant_facts_can_retry_missed_lines_with_provider(monkeypatch) -> None:
    text = "Caelan Thorne is dead."
    provider = make_provider(fallback_to_heuristic=False, retry_missed_with_ai=True)
    calls = {}

    def fake_extract_facts(text_arg, metadata_arg, provider_arg):
        calls["extract"] = (text_arg, metadata_arg, provider_arg)
        return [
            {
                "type": "status",
                "subject": "Caelan Thorne",
                "predicate": "status",
                "object": "",
                "value": "dead",
                "evidence": "Caelan Thorne is dead.",
                "story_order": 1,
            }
        ], "retry notice", ""

    monkeypatch.setattr(extraction, "extract_facts", fake_extract_facts)

    merged, notice = recover_uncovered_relevant_facts(
        text,
        [],
        {"source": "test", "chapter": 1, "story_order": 1},
        provider,
    )

    assert calls["extract"][0] == text
    assert "Coverage AI retry checked 1 missed span(s) and added 1 fact(s)" in notice
    assert "private backup reader is disabled" not in notice
    assert merged[0]["line_start"] == 1


def test_heuristic_extract_uses_value_for_affiliation_status() -> None:
    facts = heuristic_extract(
        "Seraphine Ardent's affiliation is House Ardent.",
        {"source": "test", "chapter": 1, "story_order": 1},
    )

    assert any(
        fact["type"] == "status"
        and fact["subject"] == "Seraphine Ardent"
        and fact["predicate"] == "affiliation"
        and fact["object"] == ""
        and fact["value"] == "House Ardent"
        for fact in facts
    )


def test_heuristic_extract_uses_object_for_origin_location() -> None:
    facts = heuristic_extract(
        "Liora Vale is from Bracken Parish.",
        {"source": "test", "chapter": 1, "story_order": 1},
    )

    assert any(
        fact["type"] == "location"
        and fact["subject"] == "Liora Vale"
        and fact["predicate"] == "origin"
        and fact["object"] == "Bracken Parish"
        and fact["value"] == ""
        for fact in facts
    )


def test_sample_source_and_continuation_have_deterministic_conflicts() -> None:
    root = Path(__file__).resolve().parents[1]
    source_text = (root / "examples" / "sample_source.txt").read_text(encoding="utf-8")
    continuation_text = (root / "examples" / "sample_continuation.txt").read_text(encoding="utf-8")
    memory = extraction.attach_fact_line_refs(
        source_text,
        heuristic_extract(source_text, {"source": "source", "chapter": 1, "story_order": 1}, max_facts=200),
    )
    scene = extraction.attach_fact_line_refs(
        continuation_text,
        heuristic_extract(continuation_text, {"source": "continuation", "chapter": 2, "story_order": 2}, max_facts=200),
    )

    issues = check_continuity(continuation_text, scene, memory)
    evidence = "\n".join(issue.evidence for issue in issues)

    assert len(issues) >= 8
    assert "Kael Maren's compass" in evidence
    assert "Registrar Dahl" in evidence
    assert "Sable Wren" in evidence
    assert "Ticker" in evidence
    assert "Dorien Hale" in evidence
    assert "North corridor" in evidence
    assert "The Compact" in evidence
    assert "Ashenmere" in evidence
