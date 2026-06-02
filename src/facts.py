from __future__ import annotations

import re
import uuid
from typing import Any

from src.constants import (
    FACT_TYPES,
    NAME_RE,
    RELATIONSHIP_DIMENSIONS,
    RELATIONSHIP_GENERIC_PREDICATES,
    RELATIONSHIP_ROLE_ALIASES,
    STOP_NAMES,
)
from src.utils import (
    as_text,
    canonical_relationship_type,
    clean_known_by,
    clean_name,
    confidence_value,
    to_int,
    trim_value,
    unique_preserving_order,
)


# ---------------------------------------------------------------------------
# Relationship term extraction
# ---------------------------------------------------------------------------


def relationship_terms_from_text(value: Any) -> list[str]:
    if isinstance(value, list):
        terms: list[str] = []
        for item in value:
            terms.extend(relationship_terms_from_text(item))
        return unique_preserving_order(terms)

    text = as_text(value).lower()
    if not text:
        return []

    terms = []
    occupied_spans: list[tuple[int, int]] = []
    for alias, canonical in sorted(RELATIONSHIP_ROLE_ALIASES.items(), key=lambda item: len(item[0]), reverse=True):
        pattern = re.escape(alias).replace(r"\ ", r"[\s_-]+").replace(r"\-", r"[\s_-]+")
        for match in re.finditer(rf"(?<![a-z]){pattern}(?![a-z])", text, re.IGNORECASE):
            span = match.span()
            if any(span[0] < existing[1] and existing[0] < span[1] for existing in occupied_spans):
                continue
            terms.append(canonical)
            occupied_spans.append(span)
            break
    return unique_preserving_order(terms)


def relationship_terms_from_fact(raw: dict[str, Any]) -> list[str]:
    terms: list[str] = []
    terms.extend(relationship_terms_from_text(raw.get("relation_types", [])))
    terms.extend(relationship_terms_from_text(raw.get("relation_type", "")))
    terms.extend(relationship_terms_from_text(raw.get("value", "")))

    predicate = canonical_relationship_type(raw.get("predicate", ""))
    if predicate not in RELATIONSHIP_GENERIC_PREDICATES:
        terms.extend(relationship_terms_from_text(predicate))

    return unique_preserving_order(terms)


def relationship_dimension(relation_type: str) -> str:
    return RELATIONSHIP_DIMENSIONS.get(canonical_relationship_type(relation_type), "")


def relationship_relation_type(fact: dict[str, Any]) -> str:
    terms = relationship_terms_from_fact(fact)
    if terms:
        return terms[0]
    return canonical_relationship_type(fact.get("relation_type", ""))


# ---------------------------------------------------------------------------
# Fact creation
# ---------------------------------------------------------------------------


def stable_fact_id(type_: str, subject: str, predicate: str, object_: str, value: str, metadata: dict[str, Any]) -> str:
    raw = "|".join(
        [
            type_.lower(),
            subject.lower(),
            predicate.lower(),
            object_.lower(),
            value.lower(),
            str(metadata.get("source", "")),
            str(metadata.get("story_order", metadata.get("chapter", ""))),
        ]
    )
    return str(uuid.uuid5(uuid.NAMESPACE_URL, raw))


def line_value(value: Any) -> int | None:
    number = to_int(value, 0)
    return number or None


def make_fact(
    type_: str,
    subject: str,
    predicate: str,
    metadata: dict[str, Any],
    *,
    object_: str = "",
    value: str = "",
    known_by: list[str] | None = None,
    evidence: str = "",
    confidence: float = 0.65,
) -> dict[str, Any]:
    return {
        "id": stable_fact_id(type_, subject, predicate, object_, value, metadata),
        "type": type_ if type_ in FACT_TYPES else "event",
        "subject": subject,
        "predicate": predicate,
        "object": object_,
        "value": value,
        "known_by": known_by or [],
        "chapter": int(metadata.get("chapter", 0) or 0),
        "story_order": int(metadata.get("story_order", metadata.get("chapter", 0)) or 0),
        "scene_time": str(metadata.get("scene_time", "")),
        "location": str(metadata.get("location", "")),
        "pov": str(metadata.get("pov", "")),
        "source": str(metadata.get("source", "unknown")),
        "evidence": trim_value(evidence, limit=240),
        "line_start": line_value(metadata.get("line_start")),
        "line_end": line_value(metadata.get("line_end")),
        "char_start": line_value(metadata.get("char_start")),
        "char_end": line_value(metadata.get("char_end")),
        "confidence": max(0.0, min(1.0, float(confidence))),
        "extraction": "heuristic",
    }


def make_relationship_fact(
    subject: str,
    object_: str,
    relation_type: str,
    metadata: dict[str, Any],
    *,
    evidence: str = "",
    confidence: float = 0.65,
) -> dict[str, Any]:
    canonical = canonical_relationship_type(relation_type)
    fact = make_fact(
        "relationship",
        clean_name(subject),
        canonical or "relationship_to",
        metadata,
        object_=clean_name(object_),
        value=canonical,
        evidence=evidence,
        confidence=confidence,
    )
    return with_relationship_metadata(fact)


def with_relationship_metadata(fact: dict[str, Any]) -> dict[str, Any]:
    if fact.get("type") != "relationship":
        return fact

    relation_terms = relationship_terms_from_fact(fact)
    relation_type = relation_terms[0] if relation_terms else canonical_relationship_type(fact.get("relation_type", ""))
    fact["relation_type"] = relation_type
    fact["relation_dimension"] = relationship_dimension(relation_type)
    if relation_type:
        fact["predicate"] = relation_type
        fact["value"] = relation_type
    return fact


# ---------------------------------------------------------------------------
# Fact expansion and decomposition
# ---------------------------------------------------------------------------


def expand_relationship_fact(raw: dict[str, Any]) -> list[dict[str, Any]]:
    if as_text(raw.get("type", "event")).lower().strip() != "relationship":
        return [raw]

    relation_types = relationship_terms_from_fact(raw)
    if not relation_types:
        return [raw]

    expanded = []
    for relation_type in relation_types:
        item = dict(raw)
        item.pop("id", None)
        canonical = canonical_relationship_type(relation_type)
        item["predicate"] = canonical
        item["value"] = canonical
        item["relation_type"] = canonical
        item["relation_dimension"] = relationship_dimension(canonical)
        expanded.append(item)
    return expanded


def expand_atomic_fact(raw: dict[str, Any]) -> list[dict[str, Any]]:
    timeline_parts = decompose_timeline_fact(raw)
    if timeline_parts:
        return timeline_parts
    return [raw]


def decompose_timeline_fact(raw: dict[str, Any]) -> list[dict[str, Any]]:
    if as_text(raw.get("type", "event")).lower().strip() != "timeline":
        return []

    subject = clean_name(as_text(raw.get("subject", "")))
    if not subject:
        return []

    predicate = as_text(raw.get("predicate", ""))
    object_ = clean_name(as_text(raw.get("object", "")))
    value = as_text(raw.get("value", ""))
    searchable = f"{predicate} {value}"
    parts: list[dict[str, Any]] = []

    if re.search(r"\b(?:still\s+)?(?:a\s+)?student\b", searchable, re.IGNORECASE):
        parts.append(raw_fact_with(raw, "status", subject, "role", value="student"))

    if object_ and re.search(r"\bmet\b", searchable, re.IGNORECASE):
        parts.append(raw_fact_with(raw, "event", subject, "met", object_=object_))

    involvement = involvement_status_from_phrase(searchable)
    if involvement:
        business_object, status_value = involvement
        parts.append(
            raw_fact_with(
                raw,
                "status",
                subject,
                "family_business_involvement",
                object_=business_object,
                value=status_value,
            )
        )

    return dedupe_raw_facts(parts) if len(parts) >= 2 else []


def raw_fact_with(
    raw: dict[str, Any],
    type_: str,
    subject: str,
    predicate: str,
    *,
    object_: str = "",
    value: str = "",
) -> dict[str, Any]:
    item = dict(raw)
    item.pop("id", None)
    item.pop("relation_type", None)
    item.pop("relation_types", None)
    item.pop("relation_dimension", None)
    item["type"] = type_
    item["subject"] = subject
    item["predicate"] = predicate
    item["object"] = object_
    item["value"] = value
    return item


def dedupe_raw_facts(facts: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen = set()
    deduped = []
    for fact in facts:
        key = (
            as_text(fact.get("type")).lower(),
            as_text(fact.get("subject")).lower(),
            as_text(fact.get("predicate")).lower(),
            as_text(fact.get("object")).lower(),
            as_text(fact.get("value")).lower(),
        )
        if key in seen:
            continue
        seen.add(key)
        deduped.append(fact)
    return deduped


# ---------------------------------------------------------------------------
# Business-object helpers
# ---------------------------------------------------------------------------


def involvement_status_from_phrase(phrase: str) -> tuple[str, str] | None:
    match = re.search(
        r"\b(?P<neg>not\s+)?(?:involved|active|working)\s+in\s+(?P<object>the\s+family\s+business|family\s+business|business|company|organization|organisation)\b",
        phrase,
        re.IGNORECASE,
    )
    if not match:
        return None
    value = "not involved" if match.group("neg") else "involved"
    return normalize_business_object(match.group("object")), value


def normalize_business_object(value: str) -> str:
    lowered = re.sub(r"\s+", " ", value.lower()).strip()
    if lowered in {"the family business", "family business", "business"}:
        return "family business"
    if lowered == "organisation":
        return "organization"
    return lowered


# ---------------------------------------------------------------------------
# Canonicalization and validation
# ---------------------------------------------------------------------------


def canonical_predicate(value: Any) -> str:
    text = as_text(value).lower().strip()
    text = re.sub(r"[\s-]+", "_", text)
    text = re.sub(r"[^a-z0-9_]", "", text)
    return text.strip("_")


def canonicalize_fact(fact: dict[str, Any]) -> dict[str, Any] | None:
    fact = normalize_canonical_slots(fact)
    if has_uncheckable_shape(fact):
        return None
    if has_wrong_subject_evidence(fact):
        return None
    if has_unsupported_claim_terms(fact):
        return None
    return fact


def normalize_canonical_slots(fact: dict[str, Any]) -> dict[str, Any]:
    type_ = fact.get("type", "")
    if type_ == "status":
        normalize_status_slots(fact)
    elif type_ == "relationship":
        normalize_relationship_slots(fact)
    elif type_ == "knowledge":
        normalize_knowledge_slots(fact)
    elif type_ == "event":
        fact["predicate"] = canonical_predicate(fact.get("predicate", ""))
    elif type_ == "timeline":
        fact["predicate"] = canonical_predicate(fact.get("predicate", ""))
    elif type_ in {"trait", "ability", "possession", "location", "world_rule", "character"}:
        fact["predicate"] = canonical_predicate(fact.get("predicate", ""))
    return fact


def normalize_status_slots(fact: dict[str, Any]) -> None:
    combined = f"{fact.get('predicate', '')} {fact.get('object', '')} {fact.get('value', '')}"
    involvement = involvement_status_from_phrase(combined)
    if involvement:
        fact["predicate"] = "family_business_involvement"
        fact["object"], fact["value"] = involvement
        return

    value = as_text(fact.get("value", "")).lower().strip()
    predicate = as_text(fact.get("predicate", "")).lower().strip()
    if value in {"student", "teacher", "doctor", "heir", "guard", "soldier", "servant"} or "student" in predicate:
        fact["predicate"] = "role"
        fact["value"] = value or "student"
        return

    fact["predicate"] = canonical_predicate(fact.get("predicate", "")) or "status"


def normalize_relationship_slots(fact: dict[str, Any]) -> None:
    if fact.get("relation_type"):
        return
    if fact.get("object") and fact.get("value"):
        fact["predicate"] = "relationship_state"


def normalize_knowledge_slots(fact: dict[str, Any]) -> None:
    predicate = canonical_predicate(fact.get("predicate", ""))
    if predicate in {"knows", "knew"}:
        fact["predicate"] = "knows"
    elif predicate in {"learns", "learned", "discovers", "discovered", "realizes", "realized", "found_out"}:
        fact["predicate"] = "learned"
    else:
        fact["predicate"] = predicate


def has_uncheckable_shape(fact: dict[str, Any]) -> bool:
    type_ = fact.get("type", "")
    predicate = as_text(fact.get("predicate", "")).lower()
    readable_predicate = predicate.replace("_", " ")

    if type_ == "world_rule":
        return False
    if type_ == "relationship":
        return not bool(fact.get("relation_type") or (fact.get("object") and fact.get("value")))
    if type_ == "timeline":
        return predicate not in {"before", "after", "during", "same_time_as", "story_order"}
    if type_ == "knowledge":
        return predicate not in {"knows", "learned"}
    if any(marker in f" {readable_predicate} " for marker in [" when ", " while ", " because ", " but ", " and "]):
        return True
    return len(re.findall(r"[a-zA-Z]+", readable_predicate)) > 4


def has_wrong_subject_evidence(fact: dict[str, Any]) -> bool:
    if fact.get("type") != "status":
        return False
    evidence = as_text(fact.get("evidence", ""))
    if not evidence:
        return False

    subject = as_text(fact.get("subject", "")).lower()
    subject_parts = [part.lower() for part in re.findall(r"[A-Za-z]+", subject) if len(part) > 1]
    evidence_lower = evidence.lower()
    if any(part in evidence_lower for part in subject_parts):
        return False

    if not evidence_mentions_person_reference(evidence_lower):
        return True

    evidence_names = [
        name
        for name in re.findall(NAME_RE, evidence)
        if name not in STOP_NAMES and name.lower() != subject
    ]
    return bool(evidence_names)


def evidence_mentions_person_reference(evidence: str) -> bool:
    return bool(re.search(r"\b(?:she|her|he|him|his|they|them|their|student|candidate|heir|guard|teacher|doctor|soldier|servant)\b", evidence))


def has_unsupported_claim_terms(fact: dict[str, Any]) -> bool:
    evidence = as_text(fact.get("evidence", ""))
    if not evidence or fact.get("type") == "world_rule":
        return False

    claim_terms = claim_term_roots(f"{fact.get('predicate', '')} {fact.get('value', '')}")
    if fact.get("type") not in {"relationship", "event"}:
        claim_terms.update(claim_term_roots(fact.get("object", "")))
    if not claim_terms:
        return False

    evidence_terms = claim_term_roots(evidence)
    missing = claim_terms - evidence_terms
    if fact.get("type") == "knowledge" and missing:
        return True
    return len(missing) >= 2 and len(missing) >= max(2, len(claim_terms) // 2)


def claim_term_roots(text: Any) -> set[str]:
    stop = {
        "about",
        "after",
        "before",
        "business",
        "family",
        "from",
        "involvement",
        "into",
        "known",
        "knows",
        "learned",
        "role",
        "status",
        "that",
        "their",
        "there",
        "this",
        "value",
        "with",
    }
    roots = set()
    for word in re.findall(r"[a-zA-Z]{4,}", as_text(text).lower()):
        if word in stop:
            continue
        if word.endswith("ies") and len(word) > 4:
            word = word[:-3] + "y"
        elif word.endswith("s") and len(word) > 4:
            word = word[:-1]
        roots.add(word)
    return roots


# ---------------------------------------------------------------------------
# Normalize and dedupe
# ---------------------------------------------------------------------------


def normalize_facts(
    raw_facts: list[dict[str, Any]],
    metadata: dict[str, Any],
    *,
    extraction_source: str = "import",
) -> list[dict[str, Any]]:
    normalized = []
    for raw in raw_facts:
        if not isinstance(raw, dict):
            continue

        for atomic_raw in expand_atomic_fact(raw):
            for item in expand_relationship_fact(atomic_raw):
                type_ = as_text(item.get("type", "event")).lower().strip()
                subject = clean_name(as_text(item.get("subject", "")).strip())
                predicate = as_text(item.get("predicate", "")).strip() or "mentions"
                object_ = trim_value(as_text(item.get("object", "")).strip())
                value = trim_value(as_text(item.get("value", "")).strip())
                evidence = trim_value(as_text(item.get("evidence", "")).strip(), limit=240)
                if not subject:
                    continue

                fact = {
                    "id": as_text(item.get("id")) or stable_fact_id(type_, subject, predicate, object_, value, metadata),
                    "type": type_ if type_ in FACT_TYPES else "event",
                    "subject": subject,
                    "predicate": predicate,
                    "object": object_,
                    "value": value,
                    "known_by": clean_known_by(item.get("known_by", [])),
                    "chapter": to_int(item.get("chapter", metadata.get("chapter", 0)), 0),
                    "story_order": to_int(
                        item.get("story_order", metadata.get("story_order", metadata.get("chapter", 0))),
                        0,
                    ),
                    "scene_time": as_text(item.get("scene_time", metadata.get("scene_time", ""))),
                    "location": as_text(item.get("location", metadata.get("location", ""))),
                    "pov": as_text(item.get("pov", metadata.get("pov", ""))),
                    "source": as_text(item.get("source", metadata.get("source", "unknown"))),
                    "evidence": evidence,
                    "line_start": line_value(item.get("line_start", metadata.get("line_start"))),
                    "line_end": line_value(item.get("line_end", metadata.get("line_end"))),
                    "char_start": line_value(item.get("char_start", metadata.get("char_start"))),
                    "char_end": line_value(item.get("char_end", metadata.get("char_end"))),
                    "confidence": confidence_value(item.get("confidence", 0.7)),
                    "extraction": as_text(item.get("extraction", extraction_source)),
                }
                canonical_fact = canonicalize_fact(with_relationship_metadata(fact))
                if canonical_fact:
                    normalized.append(canonical_fact)

    return dedupe_facts(normalized)


def dedupe_facts(facts: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen = set()
    deduped = []
    for fact in facts:
        key = (
            fact["type"].lower(),
            fact["subject"].lower(),
            fact["predicate"].lower(),
            fact["object"].lower(),
            fact["value"].lower(),
            fact["story_order"],
        )
        if key in seen:
            continue
        seen.add(key)
        deduped.append(fact)
    return deduped
