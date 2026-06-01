from __future__ import annotations

import re
from typing import Any

from src.constants import (
    NAME_RE,
    RELATIONSHIP_CONFLICT_PAIRS,
    ContinuityIssue,
)
from src.extraction import extract_names
from src.facts import (
    relationship_dimension,
    relationship_relation_type,
    relationship_terms_from_fact,
)
from src.utils import (
    canonical_relationship_type,
    fact_label,
    significant_words,
    text_overlap,
)


def check_continuity(
    scene_text: str,
    scene_facts: list[dict[str, Any]],
    memory: list[dict[str, Any]],
) -> list[ContinuityIssue]:
    scoped = scope_memory(scene_text, scene_facts, memory)
    issues: list[ContinuityIssue] = []

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
    return dedupe_issues(issues)


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
    comparable_types = {"relationship", "status", "trait", "ability", "possession"}
    if scene_fact["type"] not in comparable_types or memory_fact["type"] not in comparable_types:
        return None
    if scene_fact["type"] != memory_fact["type"]:
        return None
    if scene_fact["subject"].lower() != memory_fact["subject"].lower():
        return None
    if scene_fact["type"] == "relationship":
        return relationship_conflict(scene_fact, memory_fact)
    if scene_fact["predicate"].lower() != memory_fact["predicate"].lower():
        return None

    scene_slot = (scene_fact.get("object", "").lower(), scene_fact.get("value", "").lower())
    memory_slot = (memory_fact.get("object", "").lower(), memory_fact.get("value", "").lower())
    if not any(scene_slot) or not any(memory_slot) or scene_slot == memory_slot:
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
    )


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
            )
        )
    return issues


def dedupe_issues(issues: list[ContinuityIssue]) -> list[ContinuityIssue]:
    seen = set()
    deduped = []
    for issue in issues:
        key = (issue.category, issue.message, issue.evidence)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(issue)
    return deduped


def is_same_fact(a: dict[str, Any], b: dict[str, Any]) -> bool:
    return a.get("id") == b.get("id")
