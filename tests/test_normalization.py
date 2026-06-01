import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.facts import normalize_facts


def metadata() -> dict[str, object]:
    return {"source": "story note", "chapter": 1, "story_order": 1}


def test_uncheckable_knowledge_blob_is_rejected() -> None:
    raw = [
        {
            "type": "knowledge",
            "subject": "Reika",
            "predicate": "revealed her ex-boyfriend was her",
            "object": "classmate",
            "value": "classmates",
            "known_by": ["Reika", "Rikiya Anraku", "Kaito Kujou", "Maria"],
            "evidence": "We were classmates",
            "confidence": 1,
        }
    ]

    assert normalize_facts(raw, metadata(), extraction_source="gemini:gemini-3.5-flash") == []


def test_compound_timeline_fact_decomposes_to_atomic_facts() -> None:
    raw = [
        {
            "type": "timeline",
            "subject": "Sena Hikawa",
            "predicate": "was a student when she met",
            "object": "Kaito Kujou",
            "value": "not involved in family business",
            "known_by": ["Kaito Kujou", "Sena Hikawa"],
            "evidence": "At the time, she was still a student and should not have been involved in the family business at all",
            "confidence": 1,
        }
    ]

    facts = normalize_facts(raw, metadata(), extraction_source="gemini:gemini-3.5-flash")

    assert [(fact["type"], fact["subject"], fact["predicate"], fact["object"], fact["value"]) for fact in facts] == [
        ("status", "Sena Hikawa", "role", "", "student"),
        ("event", "Sena Hikawa", "met", "Kaito Kujou", ""),
        ("status", "Sena Hikawa", "family_business_involvement", "family business", "not involved"),
    ]


def test_canonical_knowledge_fact_is_kept() -> None:
    raw = [
        {
            "type": "knowledge",
            "subject": "Mara",
            "predicate": "knows",
            "object": "gate password",
            "value": "known",
            "known_by": ["Mara"],
            "evidence": "Mara knows the gate password.",
            "confidence": 0.9,
        }
    ]

    facts = normalize_facts(raw, metadata(), extraction_source="gemini:gemini-3.5-flash")

    assert len(facts) == 1
    assert facts[0]["type"] == "knowledge"
    assert facts[0]["predicate"] == "knows"
    assert facts[0]["object"] == "gate password"
