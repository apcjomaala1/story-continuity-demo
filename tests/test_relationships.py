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


def test_same_first_name_with_different_surname_warns_about_identity_drift() -> None:
    memory = normalize_facts(
        [
            {
                "type": "character",
                "subject": "Caelan Thorne",
                "predicate": "identity",
                "value": "Crown Prince of Merrowyn",
                "evidence": "Caelan Thorne is Crown Prince of Merrowyn.",
            }
        ],
        metadata(),
    )

    issues = check_continuity("Caelan Ardent was a cheerful commoner artificer.", [], memory)

    assert any(issue.category == "Possible identity drift" for issue in issues)


def test_different_named_academy_warns_about_setting_drift() -> None:
    memory = normalize_facts(
        [
            {
                "type": "location",
                "subject": "Veyrfall Academy",
                "predicate": "setting",
                "evidence": "Veyrfall Academy leaned into the sea wind.",
            }
        ],
        metadata(),
    )

    issues = check_continuity("Liora woke in the desert dormitory of Sunspire Academy.", [], memory)

    assert any(issue.category == "Possible setting drift" for issue in issues)


def test_different_possessions_do_not_conflict() -> None:
    memory = normalize_facts(
        [{"type": "possession", "subject": "Liora Vale", "predicate": "owns", "object": "iron key"}],
        metadata(),
    )
    scene = normalize_facts(
        [{"type": "possession", "subject": "Liora Vale", "predicate": "owns", "object": "trunk"}],
        metadata(),
    )

    issues = check_continuity("Liora Vale owns a trunk.", scene, memory)

    assert not [issue for issue in issues if issue.category in {"Fact conflict", "Possession conflict"}]


def test_same_status_split_between_object_and_value_does_not_conflict() -> None:
    memory = normalize_facts(
        [{"type": "status", "subject": "Seraphine Ardent", "predicate": "affiliation", "object": "House Ardent"}],
        metadata(),
    )
    scene = normalize_facts(
        [{"type": "status", "subject": "Seraphine Ardent", "predicate": "affiliation", "value": "House Ardent"}],
        metadata(),
    )

    issues = check_continuity("Seraphine Ardent's affiliation is House Ardent.", scene, memory)

    assert not [issue for issue in issues if issue.category == "Fact conflict"]
