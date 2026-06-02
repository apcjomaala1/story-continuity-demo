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


def test_same_scene_fact_conflict_warns_without_memory() -> None:
    scene = normalize_facts(
        [
            {"type": "trait", "subject": "Liora", "predicate": "hair_color", "value": "dark"},
            {"type": "trait", "subject": "Liora", "predicate": "hair_color", "value": "red"},
        ],
        metadata(),
    )

    issues = check_continuity("Liora's hair was dark. Liora's hair was red.", scene, [])

    assert any(issue.category == "Same-scene fact conflict" for issue in issues)


def test_student_and_candidate_roles_do_not_conflict() -> None:
    scene = normalize_facts(
        [
            {"type": "status", "subject": "Liora Vale", "predicate": "role", "value": "student"},
            {"type": "status", "subject": "Liora Vale", "predicate": "role", "value": "blade-veil candidate"},
        ],
        metadata(),
    )

    issues = check_continuity("Liora Vale is a student and a blade-veil candidate.", scene, [])

    assert not [issue for issue in issues if issue.category == "Same-scene fact conflict"]


def test_student_and_first_year_student_roles_do_not_conflict() -> None:
    scene = normalize_facts(
        [
            {"type": "status", "subject": "Liora Vale", "predicate": "role", "value": "student"},
            {"type": "status", "subject": "Liora Vale", "predicate": "role", "value": "first-year student"},
        ],
        metadata(),
    )

    issues = check_continuity("Liora Vale is a first-year student.", scene, [])

    assert not [issue for issue in issues if issue.category == "Same-scene fact conflict"]


def test_same_scene_identity_drift_warns_without_memory() -> None:
    issues = check_continuity("Caelan Ardent entered. Caelan Thorne followed.", [], [])

    assert any(issue.category == "Same-scene identity drift" for issue in issues)


def test_same_scene_setting_drift_warns_without_memory() -> None:
    issues = check_continuity("Liora enrolled at Sunspire Academy. Veyrfall Academy locked its gates.", [], [])

    assert any(issue.category == "Same-scene setting drift" for issue in issues)


def test_prefix_name_does_not_trigger_same_scene_identity_drift() -> None:
    """'Caelan Aer' is a prefix of 'Caelan Aer Thorne' — same person, not a drift."""
    text = "Prince Caelan Aer Thorne crossed the Tide Gate without waiting for Quill to call him."
    issues = check_continuity(text, [], [])

    assert not [issue for issue in issues if issue.category in {"Same-scene identity drift", "Possible identity drift"}]


def test_article_prefix_does_not_trigger_same_scene_setting_drift() -> None:
    """'The Tide Gate' and 'Tide Gate' are the same place — the article shouldn't split them."""
    text = "The Tide Gate stood at the center. Liora approached the Tide Gate carefully."
    issues = check_continuity(text, [], [])

    assert not [issue for issue in issues if issue.category in {"Same-scene setting drift", "Possible setting drift"}]


def test_article_prefix_observatory_does_not_trigger_setting_drift() -> None:
    """'The North Observatory' and 'North Observatory' are the same place."""
    text = "The North Observatory was sealed sixteen years ago. She looked toward North Observatory with longing."
    issues = check_continuity(text, [], [])

    assert not [issue for issue in issues if issue.category in {"Same-scene setting drift", "Possible setting drift"}]
