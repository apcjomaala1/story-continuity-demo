from __future__ import annotations

import re
from typing import Any

from src.constants import (
    NAME_RE,
    RELATIONSHIP_CONFLICT_PAIRS,
    STOP_NAMES,
    ContinuityIssue,
)
from src.extraction import extract_names
from src.facts import (
    relationship_dimension,
    relationship_relation_type,
    relationship_terms_from_fact,
)
from src.utils import (
    as_text,
    canonical_relationship_type,
    fact_label,
    significant_words,
    text_overlap,
)

PERSON_TITLE_WORDS = {
    "Crown",
    "Duke",
    "Duchess",
    "Headmaster",
    "Headmistress",
    "House",
    "King",
    "Lady",
    "Lord",
    "Master",
    "Prince",
    "Princess",
    "Professor",
    "Queen",
    "Registrar",
}
INVALID_PERSON_NAME_PARTS = {
    *STOP_NAMES,
    "First",
    "Second",
    "Third",
    "Fourth",
    "Fifth",
    "Sixth",
    "Seventh",
    "Eighth",
    "Ninth",
    "Tenth",
}
PLACE_SUFFIXES = {
    "Academy",
    "Dormitory",
    "Gate",
    "Hall",
    "Library",
    "Observatory",
    "Orchard",
    "Spire",
    "Tower",
}
PLACE_RE = re.compile(
    rf"\b([A-Z][A-Za-z]+(?:\s+[A-Z][A-Za-z]+){{0,2}}\s+(?:{'|'.join(sorted(PLACE_SUFFIXES))}))\b"
)
LEADING_ARTICLE_RE = re.compile(r"^(?:The|A|An|At|In|On|By|Near)\s+", re.IGNORECASE)
POSSESSION_PRESENT = {"has", "owns", "carries", "receives", "gets", "takes", "gains", "gained", "possesses"}
POSSESSION_ABSENT = {"loses", "lost", "gives", "gave", "sold", "destroyed"}


def check_continuity(
    scene_text: str,
    scene_facts: list[dict[str, Any]],
    memory: list[dict[str, Any]],
) -> list[ContinuityIssue]:
    scoped = scope_memory(scene_text, scene_facts, memory)
    issues: list[ContinuityIssue] = []
    issues.extend(same_scene_fact_conflicts(scene_facts))

    for scene_fact in scene_facts:
        for memory_fact in scoped:
            if is_same_fact(scene_fact, memory_fact):
                continue
            relation_issue = relationship_or_status_conflict(scene_fact, memory_fact)
            if relation_issue:
                issues.append(relation_issue)

            knowledge_issue = knowledge_timing_conflict(scene_fact, memory_fact)
            if knowledge_issue:
                issues.append(knowledge_issue)

    issues.extend(lifecycle_conflicts(scene_text, scene_facts, scoped))
    issues.extend(world_rule_conflicts(scene_text, scoped))
    issues.extend(identity_name_conflicts(scene_text, memory))
    issues.extend(named_place_conflicts(scene_text, memory))
    issues.extend(same_scene_identity_name_conflicts(scene_text))
    issues.extend(same_scene_named_place_conflicts(scene_text))
    return dedupe_issues(issues)


def same_scene_fact_conflicts(scene_facts: list[dict[str, Any]]) -> list[ContinuityIssue]:
    issues = []
    for index, first in enumerate(scene_facts):
        for second in scene_facts[index + 1 :]:
            if is_same_fact(first, second):
                continue
            issue = relationship_or_status_conflict(first, second)
            if issue:
                issues.append(
                    issue_for_same_scene_conflict(
                        issue,
                        first,
                        second,
                        "Same-scene " + issue.category[:1].lower() + issue.category[1:],
                    )
                )
    return issues


def issue_for_same_scene_conflict(
    issue: ContinuityIssue,
    first: dict[str, Any],
    second: dict[str, Any],
    category: str,
) -> ContinuityIssue:
    return ContinuityIssue(
        category=category,
        severity=issue.severity,
        message=issue.message.replace("approved memory", "another fact in this text"),
        evidence=f"Earlier: {fact_label(first)}. Later: {fact_label(second)}.",
        suggestion="In Possible story facts, edit the wrong row, delete it, or uncheck Add before approving selected facts.",
        scene_line=second.get("line_start"),
        memory_line=first.get("line_start"),
    )


def scope_memory(
    scene_text: str,
    scene_facts: list[dict[str, Any]],
    memory: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    names = {name.lower() for name in extract_names(scene_text)}
    for fact in scene_facts:
        names.add(fact["subject"].lower())
        if fact.get("object"):
            names.update(part.lower() for part in re.findall(NAME_RE, fact["object"]))

    scoped = []
    for fact in memory:
        subject_hit = fact["subject"].lower() in names
        object_hit = any(name in fact.get("object", "").lower() for name in names)
        if subject_hit or object_hit or fact["type"] == "world_rule":
            scoped.append(fact)
    return scoped


def relationship_or_status_conflict(
    scene_fact: dict[str, Any],
    memory_fact: dict[str, Any],
) -> ContinuityIssue | None:
    comparable_types = {"relationship", "status", "trait", "ability", "possession", "location"}
    if scene_fact["type"] not in comparable_types or memory_fact["type"] not in comparable_types:
        return None
    if scene_fact["type"] != memory_fact["type"]:
        return None
    if scene_fact["subject"].lower() != memory_fact["subject"].lower():
        return None
    if scene_fact["type"] == "relationship":
        return relationship_conflict(scene_fact, memory_fact)
    if scene_fact["type"] == "possession":
        return possession_conflict(scene_fact, memory_fact)
    if scene_fact["predicate"].lower() != memory_fact["predicate"].lower():
        return None

    if scene_fact["type"] == "status" and scene_fact["predicate"].lower() == "role":
        return role_status_conflict(scene_fact, memory_fact)

    scene_value = comparable_fact_value(scene_fact)
    memory_value = comparable_fact_value(memory_fact)
    if not scene_value or not memory_value or scene_value == memory_value:
        return None

    return ContinuityIssue(
        category="Fact conflict",
        severity="medium",
        message=(
            f"{scene_fact['subject']} has a new {scene_fact['type']} fact that differs from approved memory."
        ),
        evidence=(
            f"Memory: {fact_label(memory_fact)}. Scene: {fact_label(scene_fact)}."
        ),
        suggestion="Confirm whether this is a deliberate change, a flashback, or a continuity mistake.",
        scene_line=scene_fact.get("line_start"),
        memory_line=memory_fact.get("line_start"),
    )


def comparable_fact_value(fact: dict[str, Any]) -> str:
    object_ = normalize_comparison_text(fact.get("object", ""))
    value = normalize_comparison_text(fact.get("value", ""))
    if object_ and value:
        return f"{object_}|{value}"
    return value or object_


def role_status_conflict(scene_fact: dict[str, Any], memory_fact: dict[str, Any]) -> ContinuityIssue | None:
    scene_role = normalize_role_value(comparable_fact_value(scene_fact))
    memory_role = normalize_role_value(comparable_fact_value(memory_fact))
    if not scene_role or not memory_role or scene_role == memory_role:
        return None
    if role_values_compatible(scene_role, memory_role):
        return None
    return None


def role_values_compatible(first: str, second: str) -> bool:
    first_words = set(first.split())
    second_words = set(second.split())
    if first_words <= second_words or second_words <= first_words:
        return True
    if "student" in first_words and "student" in second_words:
        return True
    if "candidate" in first_words and "candidate" in second_words:
        return True
    if "candidate" in first_words and "student" in second_words:
        return True
    if "student" in first_words and "candidate" in second_words:
        return True
    return False


def normalize_role_value(value: Any) -> str:
    text = normalize_comparison_text(value).replace("-", " ")
    text = re.sub(r"\b(?:a|an|the)\b", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def normalize_comparison_text(value: Any) -> str:
    return re.sub(r"\s+", " ", as_text(value).lower()).strip(" .,:;!?\"'")


def possession_conflict(scene_fact: dict[str, Any], memory_fact: dict[str, Any]) -> ContinuityIssue | None:
    scene_item = normalize_comparison_text(scene_fact.get("object") or scene_fact.get("value"))
    memory_item = normalize_comparison_text(memory_fact.get("object") or memory_fact.get("value"))
    if not scene_item or scene_item != memory_item:
        return None

    scene_state = possession_state(scene_fact)
    memory_state = possession_state(memory_fact)
    if not scene_state or not memory_state or scene_state == memory_state:
        return None

    return ContinuityIssue(
        category="Possession conflict",
        severity="medium",
        message=f"{scene_fact['subject']}'s possession state for {scene_fact.get('object') or scene_fact.get('value')} changed.",
        evidence=f"Memory: {fact_label(memory_fact)}. Scene: {fact_label(scene_fact)}.",
        suggestion="Keep both only if the scene order shows the item was gained, lost, given away, or recovered.",
        scene_line=scene_fact.get("line_start"),
        memory_line=memory_fact.get("line_start"),
    )


def possession_state(fact: dict[str, Any]) -> str:
    predicate = normalize_comparison_text(fact.get("predicate", ""))
    if predicate in POSSESSION_ABSENT:
        return "absent"
    if predicate in POSSESSION_PRESENT:
        return "present"
    return ""


def relationship_conflict(
    scene_fact: dict[str, Any],
    memory_fact: dict[str, Any],
) -> ContinuityIssue | None:
    scene_object = scene_fact.get("object", "").lower()
    memory_object = memory_fact.get("object", "").lower()
    if not scene_object or not memory_object or scene_object != memory_object:
        return None

    scene_relation = relationship_relation_type(scene_fact)
    memory_relation = relationship_relation_type(memory_fact)
    if not scene_relation or not memory_relation or scene_relation == memory_relation:
        return None
    if not relationship_types_conflict(scene_relation, memory_relation):
        return None

    dimension = relationship_dimension(scene_relation) or "relationship"
    return ContinuityIssue(
        category="Relationship conflict",
        severity="medium",
        message=(
            f"{scene_fact['subject']} has a {dimension} relationship to {scene_fact['object']} "
            "that conflicts with approved memory."
        ),
        evidence=(
            f"Memory: {fact_label(memory_fact)}. Scene: {fact_label(scene_fact)}."
        ),
        suggestion=(
            "Keep both if the relationship changed intentionally and story order supports it; "
            "otherwise approve the correct relationship state."
        ),
        scene_line=scene_fact.get("line_start"),
        memory_line=memory_fact.get("line_start"),
    )


def relationship_types_conflict(first: str, second: str) -> bool:
    first = canonical_relationship_type(first)
    second = canonical_relationship_type(second)
    if not first or not second or first == second:
        return False
    if frozenset((first, second)) in RELATIONSHIP_CONFLICT_PAIRS:
        return True
    if relationship_dimension(first) == "romantic" and relationship_dimension(second) == "romantic":
        return is_ex_relationship(first) != is_ex_relationship(second)
    return False


def is_ex_relationship(relation_type: str) -> bool:
    return canonical_relationship_type(relation_type).startswith("ex-")


def knowledge_timing_conflict(
    scene_fact: dict[str, Any],
    memory_fact: dict[str, Any],
) -> ContinuityIssue | None:
    if scene_fact["type"] != "knowledge" or memory_fact["type"] != "knowledge":
        return None
    if scene_fact["subject"].lower() != memory_fact["subject"].lower():
        return None
    if not text_overlap(scene_fact.get("object", ""), memory_fact.get("object", "")):
        return None
    if scene_fact["story_order"] >= memory_fact["story_order"]:
        return None

    return ContinuityIssue(
        category="Possible knowledge leak",
        severity="high",
        message=(
            f"{scene_fact['subject']} appears to know something before the approved reveal point."
        ),
        evidence=(
            f"Approved reveal at story order {memory_fact['story_order']}: {memory_fact.get('evidence', '')} "
            f"Scene at story order {scene_fact['story_order']}: {scene_fact.get('evidence', '')}"
        ),
        suggestion="Move the reveal earlier in memory, lower the scene story order, or rewrite the scene so the character is guessing.",
        scene_line=scene_fact.get("line_start"),
        memory_line=memory_fact.get("line_start"),
    )


def lifecycle_conflicts(
    scene_text: str,
    scene_facts: list[dict[str, Any]],
    memory: list[dict[str, Any]],
) -> list[ContinuityIssue]:
    issues = []
    scene_order = min([fact["story_order"] for fact in scene_facts], default=0)
    acting_subjects = {fact["subject"].lower() for fact in scene_facts if fact["type"] in {"event", "relationship", "possession"}}
    text_lower = scene_text.lower()

    for fact in memory:
        if fact["type"] != "status":
            continue
        value = fact.get("value", "").lower()
        if value not in {"dead", "destroyed", "missing", "sealed", "exiled"}:
            continue
        subject = fact["subject"].lower()
        if scene_order and fact["story_order"] and scene_order < fact["story_order"]:
            continue
        appears_active = subject in acting_subjects or subject in text_lower
        if not appears_active:
            continue
        issues.append(
            ContinuityIssue(
                category="Lifecycle risk",
                severity="high" if value in {"dead", "destroyed"} else "medium",
                message=f"{fact['subject']} is marked as {value} in memory but appears in the scene.",
                evidence=f"Memory: {fact.get('evidence', fact_label(fact))}",
                suggestion="Check whether this is a flashback, resurrection/repair, mistaken identity, or a real continuity issue.",
                memory_line=fact.get("line_start"),
            )
        )
    return issues


def world_rule_conflicts(scene_text: str, memory: list[dict[str, Any]]) -> list[ContinuityIssue]:
    issues = []
    lowered = scene_text.lower()
    for fact in memory:
        if fact["type"] != "world_rule":
            continue
        rule_text = f"{fact.get('value', '')} {fact.get('evidence', '')}".lower()
        if not any(word in rule_text for word in ["cannot", "can't", "never", "impossible"]):
            continue
        keywords = significant_words(rule_text)
        overlap = [word for word in keywords if word in lowered]
        if len(overlap) < 2:
            continue
        issues.append(
            ContinuityIssue(
                category="World rule risk",
                severity="low",
                message="The scene touches a topic governed by a restrictive world rule.",
                evidence=f"Rule: {fact.get('evidence', fact.get('value', ''))}",
                suggestion="Ask whether the scene violates the rule, creates a valid exception, or needs clearer wording.",
                memory_line=fact.get("line_start"),
            )
        )
    return issues


def identity_name_conflicts(scene_text: str, memory: list[dict[str, Any]]) -> list[ContinuityIssue]:
    memory_names = indexed_person_names(memory_texts(memory))
    scene_names = indexed_person_names([(scene_text, scene_text, None)])
    issues = []

    for first_name, scene_variants in scene_names.items():
        memory_variants = memory_names.get(first_name, {})
        for scene_last, scene_entry in scene_variants.items():
            for memory_last, memory_entry in memory_variants.items():
                if scene_last == memory_last:
                    continue
                issues.append(
                    ContinuityIssue(
                        category="Possible identity drift",
                        severity="medium",
                        message=(
                            f"{scene_entry['name']} uses a different family/name marker than approved memory "
                            f"for {memory_entry['name']}."
                        ),
                        evidence=(
                            f"Memory: {memory_entry['evidence']} Scene: {scene_entry['evidence']}"
                        ),
                        suggestion=(
                            "Check whether this is a rename, alias, title change, mistaken identity, "
                            "or an unintended continuity error."
                        ),
                        scene_line=scene_entry["line"],
                        memory_line=memory_entry["line"],
                    )
                )
    return issues


def same_scene_identity_name_conflicts(scene_text: str) -> list[ContinuityIssue]:
    scene_names = indexed_person_names([(scene_text, scene_text, None)])
    issues = []
    for first_name, variants in scene_names.items():
        entries = list(variants.values())
        for index, first in enumerate(entries):
            for second in entries[index + 1 :]:
                issues.append(
                    ContinuityIssue(
                        category="Same-scene identity drift",
                        severity="medium",
                        message=f"{first['name']} and {second['name']} share a first name but use different family/name markers.",
                        evidence=f"Earlier: {first['evidence']} Later: {second['evidence']}",
                        suggestion="Check whether this text intentionally uses an alias/name change or accidentally changes the character identity.",
                        scene_line=second["line"],
                        memory_line=first["line"],
                    )
                )
    return issues


def named_place_conflicts(scene_text: str, memory: list[dict[str, Any]]) -> list[ContinuityIssue]:
    memory_places = indexed_places(memory_texts(memory))
    scene_places = indexed_places([(scene_text, scene_text, None)])
    issues = []

    for suffix, scene_variants in scene_places.items():
        memory_variants = memory_places.get(suffix, {})
        for scene_key, scene_entry in scene_variants.items():
            for memory_key, memory_entry in memory_variants.items():
                if scene_key == memory_key:
                    continue
                issues.append(
                    ContinuityIssue(
                        category="Possible setting drift",
                        severity="medium",
                        message=(
                            f"The scene references {scene_entry['name']}, but approved memory references "
                            f"{memory_entry['name']} as a named {suffix.lower()}."
                        ),
                        evidence=f"Memory: {memory_entry['evidence']} Scene: {scene_entry['evidence']}",
                        suggestion=(
                            "Confirm whether this is a new location, a renamed location, a flashback, "
                            "or a continuity mistake."
                        ),
                        scene_line=scene_entry["line"],
                        memory_line=memory_entry["line"],
                    )
                )
    return issues


def same_scene_named_place_conflicts(scene_text: str) -> list[ContinuityIssue]:
    scene_places = indexed_places([(scene_text, scene_text, None)])
    issues = []
    for suffix, variants in scene_places.items():
        entries = list(variants.values())
        for index, first in enumerate(entries):
            for second in entries[index + 1 :]:
                issues.append(
                    ContinuityIssue(
                        category="Same-scene setting drift",
                        severity="medium",
                        message=f"The text references both {first['name']} and {second['name']} as named {pluralize_suffix(suffix)}.",
                        evidence=f"Earlier: {first['evidence']} Later: {second['evidence']}",
                        suggestion="Check whether this is a new location, a renamed location, or an accidental setting change inside the same text.",
                        scene_line=second["line"],
                        memory_line=first["line"],
                    )
                )
    return issues


def pluralize_suffix(value: str) -> str:
    if value.endswith("y"):
        return value[:-1] + "ies"
    return value + "s"


def memory_texts(memory: list[dict[str, Any]]) -> list[tuple[str, str, int | None]]:
    texts = []
    for fact in memory:
        evidence = fact.get("evidence") or fact_label(fact)
        text = ". ".join(
            str(part)
            for part in [
                fact.get("subject", ""),
                fact.get("predicate", ""),
                fact.get("object", ""),
                fact.get("value", ""),
                evidence,
            ]
            if part
        )
        texts.append((text, str(evidence), fact.get("line_start")))
    return texts


def indexed_person_names(texts: list[tuple[str, str, int | None]]) -> dict[str, dict[str, dict[str, Any]]]:
    indexed: dict[str, dict[str, dict[str, str]]] = {}
    for text, evidence, line_hint in texts:
        for raw_name in extract_names(text):
            name_parts = person_name_parts(raw_name)
            if not name_parts:
                continue
            first_name, last_name, display_name = name_parts
            indexed.setdefault(first_name.lower(), {}).setdefault(
                last_name.lower(),
                {
                    "name": display_name,
                    "evidence": evidence_excerpt(evidence, display_name),
                    "line": line_hint or line_number_for_snippet(text, display_name),
                },
            )
    return _merge_prefix_name_variants(indexed)


def _merge_prefix_name_variants(
    indexed: dict[str, dict[str, dict[str, Any]]],
) -> dict[str, dict[str, dict[str, Any]]]:
    """Merge name variants where a shorter name is a prefix of a longer one.

    For example, if we see both "Caelan Aer" (first=caelan, last=aer) and
    "Caelan Aer Thorne" (first=caelan, last=thorne), the shorter name is a
    substring of the longer display name, so they refer to the same person.
    Keep only the longest variant to avoid false identity-drift alerts.
    """
    merged: dict[str, dict[str, dict[str, Any]]] = {}
    for first_name, variants in indexed.items():
        if len(variants) <= 1:
            merged[first_name] = variants
            continue
        # Sort variants by display-name length descending so longer names come first.
        sorted_keys = sorted(variants, key=lambda k: len(variants[k]["name"]), reverse=True)
        kept: dict[str, dict[str, Any]] = {}
        for key in sorted_keys:
            entry = variants[key]
            display_lower = entry["name"].lower()
            # Check if this name is a prefix/substring of any already-kept longer name.
            is_prefix = any(
                display_lower in kept_entry["name"].lower()
                for kept_entry in kept.values()
            )
            if not is_prefix:
                kept[key] = entry
        merged[first_name] = kept
    return merged


def person_name_parts(name: str) -> tuple[str, str, str] | None:
    parts = [part for part in name.split() if part not in PERSON_TITLE_WORDS]
    if len(parts) < 2:
        return None
    if any(part in INVALID_PERSON_NAME_PARTS for part in parts):
        return None
    if parts[-1] in PLACE_SUFFIXES:
        return None
    if any(part.isupper() and len(part) > 1 for part in parts):
        return None
    return parts[0], parts[-1], " ".join(parts)


def indexed_places(texts: list[tuple[str, str, int | None]]) -> dict[str, dict[str, dict[str, Any]]]:
    indexed: dict[str, dict[str, dict[str, Any]]] = {}
    for text, evidence, line_hint in texts:
        for match in PLACE_RE.finditer(text):
            raw_place = match.group(1)
            # Strip leading articles so "The Tide Gate" and "Tide Gate" unify.
            place = LEADING_ARTICLE_RE.sub("", raw_place)
            parts = place.split()
            if not parts:
                continue
            if any(part.isupper() and len(part) > 1 for part in parts):
                continue
            suffix = parts[-1].lower()
            indexed.setdefault(suffix, {}).setdefault(
                place.lower(),
                {"name": place, "evidence": evidence_excerpt(evidence, raw_place), "line": line_hint or line_number_for_snippet(text, raw_place)},
            )
    return indexed


def line_number_for_snippet(text: str, needle: str) -> int | None:
    position = str(text).lower().find(str(needle).lower())
    if position < 0:
        return None
    return str(text)[:position].count("\n") + 1


def evidence_excerpt(text: str, needle: str, *, radius: int = 90) -> str:
    normalized = re.sub(r"\s+", " ", str(text)).strip()
    if not normalized:
        return needle
    position = normalized.lower().find(needle.lower())
    if position < 0:
        return normalized[:200]
    start = max(0, position - radius)
    end = min(len(normalized), position + len(needle) + radius)
    prefix = "..." if start else ""
    suffix = "..." if end < len(normalized) else ""
    return f"{prefix}{normalized[start:end]}{suffix}"


def dedupe_issues(issues: list[ContinuityIssue]) -> list[ContinuityIssue]:
    seen_exact: set[tuple[str, str, str]] = set()
    seen_entity_pairs: set[tuple[str, ...]] = set()
    deduped = []
    for issue in issues:
        exact_key = (issue.category, issue.message, issue.evidence)
        if exact_key in seen_exact:
            continue
        seen_exact.add(exact_key)
        # Detect when two issues reference the same pair of named entities
        # across different check categories (e.g. "Possible setting drift" and
        # "Same-scene setting drift" for the same Sunspire↔Veyrfall pair).
        entity_key = _issue_entity_pair_key(issue)
        if entity_key and entity_key in seen_entity_pairs:
            continue
        if entity_key:
            seen_entity_pairs.add(entity_key)
        deduped.append(issue)
    return deduped


def _issue_entity_pair_key(issue: ContinuityIssue) -> tuple[str, ...] | None:
    """Extract a normalized entity-pair key from a drift issue message.

    For messages like 'The scene references Sunspire Academy, but approved
    memory references Veyrfall Academy as a named academy' or 'Mara Auric and
    Mara Venn share a first name...', pull out the two entity names so that
    the same pair isn't reported twice from different check paths.
    """
    base = issue.category.lower().replace("same-scene ", "").replace("possible ", "")
    if base not in ("setting drift", "identity drift"):
        return None
    # Try "X and Y share" pattern (same-scene identity drift)
    m = re.match(r"^(.+?)\s+and\s+(.+?)\s+share\b", issue.message)
    if m:
        a, b = m.group(1).strip().lower(), m.group(2).strip().lower()
        return (base, *sorted([a, b]))
    # Try "references X, but ... references Y" pattern (memory drift)
    m = re.match(r".*references\s+(.+?),\s+but.*references\s+(.+?)\s+as\b", issue.message)
    if m:
        a, b = m.group(1).strip().lower(), m.group(2).strip().lower()
        return (base, *sorted([a, b]))
    return None


def is_same_fact(a: dict[str, Any], b: dict[str, Any]) -> bool:
    return a.get("id") == b.get("id")
