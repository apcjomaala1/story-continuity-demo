import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.continuity import check_continuity
from src.extraction import heuristic_extract
from src.facts import normalize_facts


def metadata() -> dict[str, object]:
    return {"source": "test", "chapter": 1, "story_order": 1}


def relationship_facts(facts: list[dict[str, object]]) -> list[dict[str, object]]:
    return [fact for fact in facts if fact["type"] == "relationship"]


def test_compound_relationship_roles_are_split() -> None:
    text = "Mara is Dain's lover and boss."

    facts = relationship_facts(heuristic_extract(text, metadata()))

    assert {
        (fact["subject"], fact["object"], fact["relation_type"], fact["relation_dimension"])
        for fact in facts
    } == {
        ("Mara", "Dain", "lover", "romantic"),
        ("Mara", "Dain", "boss", "authority"),
    }


def test_relationship_dimensions_do_not_conflict() -> None:
    memory = normalize_facts(
        [{"type": "relationship", "subject": "Mara", "predicate": "relationship_to", "object": "Dain", "value": "lover"}],
        metadata(),
    )
    scene = normalize_facts(
        [{"type": "relationship", "subject": "Mara", "predicate": "relationship_to", "object": "Dain", "value": "boss"}],
        metadata(),
    )

    issues = check_continuity("Mara is Dain's boss.", scene, memory)

    assert not [issue for issue in issues if issue.category == "Relationship conflict"]


def test_longer_relationship_roles_do_not_create_nested_roles() -> None:
    facts = normalize_facts(
        [
            {
                "type": "relationship",
                "subject": "Mara",
                "predicate": "relationship_to",
                "object": "Dain",
                "value": "ex-girlfriend",
            }
        ],
        metadata(),
    )

    assert [(fact["relation_type"], fact["relation_dimension"]) for fact in facts] == [("ex-girlfriend", "romantic")]


def test_relationship_value_is_not_assumed_to_be_a_role() -> None:
    facts = normalize_facts(
        [
            {
                "type": "relationship",
                "subject": "Mara",
                "predicate": "relationship_to",
                "object": "Dain",
                "value": "strained",
            }
        ],
        metadata(),
    )

    assert facts[0]["relation_type"] == ""
    assert facts[0]["value"] == "strained"


def test_conflicting_relationship_status_still_warns() -> None:
    memory = normalize_facts(
        [{"type": "relationship", "subject": "Mara", "predicate": "relationship_to", "object": "Dain", "value": "wife"}],
        metadata(),
    )
    scene = normalize_facts(
        [{"type": "relationship", "subject": "Mara", "predicate": "relationship_to", "object": "Dain", "value": "girlfriend"}],
        metadata(),
    )

    issues = check_continuity("Mara introduced herself as Dain's girlfriend.", scene, memory)

    assert any(issue.category == "Relationship conflict" for issue in issues)
