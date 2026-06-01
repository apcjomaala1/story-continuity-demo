from __future__ import annotations

import re
from typing import Any

from src.constants import RELATIONSHIP_ROLE_ALIASES


def as_text(value: Any, default: str = "") -> str:
    if value is None:
        return default
    if isinstance(value, str):
        return value
    if isinstance(value, (int, float, bool)):
        return str(value)
    return default


def to_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def confidence_value(value: Any) -> float:
    try:
        confidence = float(value)
    except (TypeError, ValueError):
        confidence = 0.7
    return max(0.0, min(1.0, confidence))


def clean_name(value: str) -> str:
    cleaned = re.sub(r"\s+", " ", value).strip(" ,.;:!?\"'()[]")
    return cleaned


def trim_value(value: str, limit: int = 180) -> str:
    cleaned = re.sub(r"\s+", " ", value).strip(" ,.;:!?\"'()[]")
    if len(cleaned) <= limit:
        return cleaned
    return cleaned[: limit - 3].rstrip() + "..."


def clean_known_by(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [clean_name(as_text(item)) for item in value if clean_name(as_text(item))]


def significant_words(text: str) -> set[str]:
    stop = {"the", "and", "that", "with", "from", "this", "there", "their", "about", "into", "after", "before"}
    return {word for word in re.findall(r"[a-zA-Z]{4,}", text.lower()) if word not in stop}


def text_overlap(a: str, b: str) -> bool:
    a_words = significant_words(a.lower())
    b_words = significant_words(b.lower())
    return len(a_words.intersection(b_words)) >= 2


def canonical_relationship_type(value: Any) -> str:
    text = as_text(value).lower().strip()
    text = re.sub(r"[\s_]+", " ", text)
    text = text.strip(" ,.;:!?\"'()[]")
    if not text:
        return ""
    return RELATIONSHIP_ROLE_ALIASES.get(text, text.replace(" ", "-"))


def unique_preserving_order(values: list[str]) -> list[str]:
    seen = set()
    unique = []
    for value in values:
        normalized = canonical_relationship_type(value)
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        unique.append(normalized)
    return unique


def sorted_unique(values: list[str]) -> list[str]:
    seen = set()
    unique = []
    for value in values:
        normalized = value.strip()
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        unique.append(normalized)
    return sorted(unique, key=str.lower)


def summarize_exception(exc: Exception) -> str:
    message = str(exc).strip()
    if len(message) > 180:
        message = message[:177].rstrip() + "..."
    return f"{type(exc).__name__}: {message}"


def truncate_debug_text(value: str, limit: int) -> str:
    if len(value) <= limit:
        return value
    omitted = len(value) - limit
    return f"{value[:limit]}\n\n... <truncated {omitted} chars>"


def humanize_token(value: Any) -> str:
    text = as_text(value).strip()
    text = text.replace("_", " ").replace("-", " ")
    return re.sub(r"\s+", " ", text).strip()


def possessive_name(value: str) -> str:
    if not value:
        return "This"
    return f"{value}'" if value.endswith("s") else f"{value}'s"


def fact_label(fact: dict[str, Any]) -> str:
    from src.facts import relationship_relation_type

    subject = as_text(fact.get("subject"))
    predicate = humanize_token(fact.get("predicate"))
    object_ = as_text(fact.get("object"))
    value = as_text(fact.get("value"))

    if fact.get("type") == "relationship":
        relation_type = relationship_relation_type(fact)
        relation = humanize_token(relation_type)
        if subject and object_ and relation:
            return f"{subject} is {object_}'s {relation}."
        parts = ["Relationship", subject]
        if relation_type:
            parts.append(relation)
        if object_:
            parts.append(object_)
        if value and canonical_relationship_type(value) != relation_type:
            parts.append(f"= {value}")
        return " | ".join(part for part in parts if part)

    fact_type = as_text(fact.get("type"), "fact")
    if fact_type == "knowledge" and subject and object_:
        return f"{subject} knows that {object_}."
    if fact_type == "status" and subject and predicate and value:
        return f"{possessive_name(subject)} {predicate} is {value}."
    if fact_type == "trait" and subject and predicate and value:
        return f"{possessive_name(subject)} {predicate} is {value}."
    if fact_type == "ability" and subject and value:
        return f"{subject} has ability: {value}."
    if fact_type == "location" and subject and (object_ or value):
        place = object_ or value
        return f"{subject} is at {place}."
    if fact_type == "possession" and subject and object_:
        action = predicate or "has"
        return f"{subject} {action} {object_}."
    if fact_type == "event" and subject and predicate:
        tail = f" {object_}" if object_ else ""
        detail = f": {value}" if value and value != object_ else ""
        return f"{subject} {predicate}{tail}{detail}."
    if fact_type == "world_rule":
        rule = value or object_ or subject
        return f"World rule: {rule}."
    if fact_type == "timeline" and subject and predicate:
        tail = f" {object_}" if object_ else ""
        return f"{subject} {predicate}{tail}."

    parts = [humanize_token(fact_type), subject]
    if predicate:
        parts.append(predicate)
    if object_:
        parts.append(object_)
    if value:
        parts.append(f"= {value}")
    label = " | ".join(part for part in parts if part)
    return label[:1].upper() + label[1:] if label else "Fact"
