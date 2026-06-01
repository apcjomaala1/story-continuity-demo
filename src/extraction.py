from __future__ import annotations

import re
from typing import Any

from src.constants import (
    NAME_RE,
    RELATIONSHIP_ACTIONS,
    RELATIONSHIP_ROLE_LIST_PATTERN,
    STOP_NAMES,
    ProviderConfig,
)
from src.facts import (
    dedupe_facts,
    make_fact,
    make_relationship_fact,
    normalize_business_object,
    normalize_facts,
    relationship_terms_from_text,
)
from src.providers import (
    build_provider_debug,
    call_claude,
    call_gemini,
    call_ollama,
    call_openai_compatible,
    parse_json_object,
    provider_source_label,
    suggested_char_limit,
)
from src.utils import clean_name, summarize_exception, trim_value


# ---------------------------------------------------------------------------
# Top-level extraction orchestration
# ---------------------------------------------------------------------------


def extract_facts_for_text(
    text: str,
    metadata: dict[str, Any],
    provider: ProviderConfig,
    *,
    split_over_limit: bool,
) -> tuple[list[dict[str, Any]], str, str]:
    limit = suggested_char_limit(provider)
    chunks = split_text_into_chunks(text, limit) if split_over_limit and len(text) > limit else [text]
    if len(chunks) == 1:
        return extract_facts(text, metadata, provider)

    all_facts: list[dict[str, Any]] = []
    debug_blocks: list[str] = []
    fallback_count = 0
    failed_without_fallback = 0

    for index, chunk in enumerate(chunks, start=1):
        chunk_metadata = {
            **metadata,
            "chunk": f"{index}/{len(chunks)}",
            "chunk_chars": len(chunk),
        }
        facts, notice, debug = extract_facts(chunk, chunk_metadata, provider)
        all_facts.extend(facts)

        if "Fell back to private heuristic" in notice:
            fallback_count += 1
        if "No fallback was used" in notice:
            failed_without_fallback += 1
        if debug:
            debug_blocks.append(f"--- Chunk {index}/{len(chunks)} ({len(chunk):,} chars) ---\n{debug}")

    merged_facts = dedupe_facts(all_facts)
    notice = (
        f"Split text into {len(chunks)} parts under {limit:,} chars and made "
        f"{len(chunks)} extraction calls. Extracted {len(merged_facts)} deduped fact(s)."
    )
    if fallback_count:
        notice += f" {fallback_count} part(s) used private heuristic fallback."
    if failed_without_fallback:
        notice += f" {failed_without_fallback} part(s) failed without fallback."

    return merged_facts, notice, "\n\n".join(debug_blocks)


def extract_facts(
    text: str,
    metadata: dict[str, Any],
    provider: ProviderConfig,
) -> tuple[list[dict[str, Any]], str, str]:
    raw = ""
    source = provider_source_label(provider)
    try:
        if provider.mode == "Ollama local LLM":
            raw = call_ollama(text, metadata, provider)
        elif provider.mode == "Gemini API":
            raw = call_gemini(text, metadata, provider)
        elif provider.mode == "Claude API":
            raw = call_claude(text, metadata, provider)
        else:
            raw = call_openai_compatible(text, metadata, provider)

        payload = parse_json_object(raw)
        raw_facts = payload.get("facts", [])
        facts = normalize_facts(raw_facts, metadata, extraction_source=source)
        raw_count = len(raw_facts) if isinstance(raw_facts, list) else 0
        return facts, f"Used {source}; normalized {raw_count} provider item(s) into {len(facts)} canonical fact(s).", ""
    except Exception as exc:
        debug = build_provider_debug(exc, provider, metadata, text, raw)
        if not provider.fallback_to_heuristic:
            return [], f"Provider failed ({summarize_exception(exc)}). No fallback was used.", debug
        facts = heuristic_extract(text, metadata)
        return facts, f"Provider failed ({summarize_exception(exc)}). Fell back to private heuristic extraction.", debug


# ---------------------------------------------------------------------------
# Text chunking
# ---------------------------------------------------------------------------


def split_text_into_chunks(text: str, limit: int) -> list[str]:
    normalized = text.replace("\r\n", "\n").replace("\r", "\n")
    if limit <= 0 or len(normalized) <= limit:
        return [normalized]

    part_count = (len(normalized) + limit - 1) // limit
    natural_boundaries = find_chunk_boundaries(normalized, sentence_only=False)
    sentence_boundaries = find_chunk_boundaries(normalized, sentence_only=True)
    cuts = choose_chunk_cuts(normalized, limit, part_count, natural_boundaries, sentence_boundaries)

    chunks = []
    start = 0
    for cut in cuts + [len(normalized)]:
        chunk = normalized[start:cut].strip()
        if chunk:
            chunks.append(chunk)
        start = cut

    if all(len(chunk) <= limit for chunk in chunks):
        return chunks
    return split_oversized_chunks(chunks, limit)


def find_chunk_boundaries(text: str, *, sentence_only: bool) -> list[int]:
    boundaries = set()
    for match in re.finditer(r"\n+", text):
        boundaries.add(match.end())
    for match in re.finditer(r"(?<=[.!?])(?:[\"')\]]+)?\s+", text):
        boundaries.add(match.end())
    if not sentence_only:
        for match in re.finditer(r"\s+", text):
            boundaries.add(match.end())
    return sorted(position for position in boundaries if 0 < position < len(text))


def choose_chunk_cuts(
    text: str,
    limit: int,
    part_count: int,
    natural_boundaries: list[int],
    sentence_boundaries: list[int],
) -> list[int]:
    cuts = []
    previous = 0
    total = len(text)

    for part_index in range(1, part_count):
        remaining_parts = part_count - part_index
        ideal = round(total * part_index / part_count)
        min_cut = max(previous + 1, total - remaining_parts * limit)
        max_cut = min(previous + limit, total - remaining_parts)
        cut = nearest_boundary(sentence_boundaries, ideal, min_cut, max_cut)
        if cut is None:
            cut = nearest_boundary(natural_boundaries, ideal, min_cut, max_cut)
        if cut is None:
            cut = max_cut
        cuts.append(cut)
        previous = cut

    return cuts


def nearest_boundary(boundaries: list[int], target: int, minimum: int, maximum: int) -> int | None:
    candidates = [position for position in boundaries if minimum <= position <= maximum]
    if not candidates:
        return None
    return min(candidates, key=lambda position: (abs(position - target), position))


def split_oversized_chunks(chunks: list[str], limit: int) -> list[str]:
    final_chunks = []
    for chunk in chunks:
        if len(chunk) <= limit:
            final_chunks.append(chunk)
            continue
        start = 0
        while start < len(chunk):
            cut = min(start + limit, len(chunk))
            if cut < len(chunk):
                space = chunk.rfind(" ", start, cut)
                if space > start:
                    cut = space + 1
            final_chunks.append(chunk[start:cut].strip())
            start = cut
    return [chunk for chunk in final_chunks if chunk]


# ---------------------------------------------------------------------------
# Heuristic extraction (regex-based fallback)
# ---------------------------------------------------------------------------


def heuristic_extract(text: str, metadata: dict[str, Any]) -> list[dict[str, Any]]:
    sentences = split_sentences(text)
    facts: list[dict[str, Any]] = []

    for sentence in sentences:
        sentence_meta = metadata_for_sentence(sentence, metadata)
        facts.extend(extract_relationships(sentence, sentence_meta))
        facts.extend(extract_knowledge(sentence, sentence_meta))
        facts.extend(extract_status(sentence, sentence_meta))
        facts.extend(extract_possessions(sentence, sentence_meta))
        facts.extend(extract_world_rules(sentence, sentence_meta))
        facts.extend(extract_events(sentence, sentence_meta))

    return dedupe_facts(facts)[:24]


def metadata_for_sentence(sentence: str, metadata: dict[str, Any]) -> dict[str, Any]:
    sentence_meta = dict(metadata)
    chapter_match = re.search(r"\bchapter\s+(?P<chapter>\d+)\b", sentence, re.IGNORECASE)
    if chapter_match:
        chapter = int(chapter_match.group("chapter"))
        sentence_meta["chapter"] = chapter
        sentence_meta["story_order"] = chapter
    return sentence_meta


def split_sentences(text: str) -> list[str]:
    normalized = re.sub(r"\s+", " ", text.replace("\n", " ")).strip()
    if not normalized:
        return []
    return [
        sentence.strip()
        for sentence in re.split(r"(?<=[.!?])\s+", normalized)
        if len(sentence.strip()) > 8
    ]


def extract_relationships(sentence: str, metadata: dict[str, Any]) -> list[dict[str, Any]]:
    facts: list[dict[str, Any]] = []
    possessive = re.compile(
        rf"\b(?P<a>{NAME_RE})\s+(?:is|was|became|becomes|remains)\s+(?:both\s+)?(?P<b>{NAME_RE})'s\s+(?P<roles>{RELATIONSHIP_ROLE_LIST_PATTERN})\b",
        re.IGNORECASE,
    )
    self_label = re.compile(
        rf"\b(?P<a>{NAME_RE})\s+(?:introduced\s+(?:herself|himself|themself)\s+as|called\s+(?:herself|himself|themself))\s+(?:both\s+)?(?P<b>{NAME_RE})'s\s+(?P<roles>{RELATIONSHIP_ROLE_LIST_PATTERN})\b",
        re.IGNORECASE,
    )
    action = re.compile(
        rf"\b(?P<a>{NAME_RE})\s+(?P<verb>{'|'.join(RELATIONSHIP_ACTIONS)})\s+(?P<b>{NAME_RE})\b",
        re.IGNORECASE,
    )

    for match in list(possessive.finditer(sentence)) + list(self_label.finditer(sentence)):
        for role in relationship_terms_from_text(match.group("roles")):
            facts.append(
                make_relationship_fact(
                    match.group("a"),
                    match.group("b"),
                    role,
                    metadata,
                    evidence=sentence,
                )
            )

    for match in action.finditer(sentence):
        facts.append(
            make_relationship_fact(
                match.group("a"),
                match.group("b"),
                match.group("verb"),
                metadata,
                evidence=sentence,
            )
        )

    return facts


def extract_knowledge(sentence: str, metadata: dict[str, Any]) -> list[dict[str, Any]]:
    facts: list[dict[str, Any]] = []
    pattern = re.compile(
        rf"\b(?P<a>{NAME_RE})\s+(?P<verb>knows|knew|learns|learned|discovers|discovered|realizes|realized|found out)\s+(?:that\s+)?(?P<object>[^.!?]+)",
        re.IGNORECASE,
    )
    for match in pattern.finditer(sentence):
        facts.append(
            make_fact(
                "knowledge",
                clean_name(match.group("a")),
                match.group("verb").lower(),
                metadata,
                object_=trim_value(match.group("object")),
                value="known",
                known_by=[clean_name(match.group("a"))],
                evidence=sentence,
            )
        )
    return facts


def extract_status(sentence: str, metadata: dict[str, Any]) -> list[dict[str, Any]]:
    facts: list[dict[str, Any]] = []
    state_words = "dead|alive|missing|destroyed|broken|married|single|injured|blind|pregnant|exiled|sealed|hidden"
    pattern = re.compile(
        rf"\b(?P<a>{NAME_RE})\s+(?:is|was|becomes|became|remains|stays)\s+(?P<state>{state_words})\b",
        re.IGNORECASE,
    )
    role_pattern = re.compile(
        rf"\b(?P<a>{NAME_RE})\s+(?:is|was|becomes|became|remains|stays)\s+(?:still\s+)?(?:a|an)\s+(?P<role>student|teacher|doctor|heir|guard|soldier|servant)\b",
        re.IGNORECASE,
    )
    involvement_pattern = re.compile(
        rf"\b(?P<a>{NAME_RE})\s+(?:is|was|becomes|became|remains|stays)\s+(?P<neg>not\s+)?(?:involved|active|working)\s+in\s+(?P<object>family business|the family business|business|company|organization|organisation)\b",
        re.IGNORECASE,
    )
    death_pattern = re.compile(rf"\b(?P<a>{NAME_RE})\s+(?:died|dies|was killed|gets killed)\b", re.IGNORECASE)

    for match in pattern.finditer(sentence):
        facts.append(
            make_fact(
                "status",
                clean_name(match.group("a")),
                "status",
                metadata,
                value=match.group("state").lower(),
                evidence=sentence,
            )
        )
    for match in role_pattern.finditer(sentence):
        facts.append(
            make_fact(
                "status",
                clean_name(match.group("a")),
                "role",
                metadata,
                value=match.group("role").lower(),
                evidence=sentence,
            )
        )
    for match in involvement_pattern.finditer(sentence):
        status_value = "not involved" if match.group("neg") else "involved"
        facts.append(
            make_fact(
                "status",
                clean_name(match.group("a")),
                "family_business_involvement",
                metadata,
                object_=normalize_business_object(match.group("object")),
                value=status_value,
                evidence=sentence,
            )
        )
    for match in death_pattern.finditer(sentence):
        facts.append(
            make_fact(
                "status",
                clean_name(match.group("a")),
                "status",
                metadata,
                value="dead",
                evidence=sentence,
            )
        )
    return facts


def extract_possessions(sentence: str, metadata: dict[str, Any]) -> list[dict[str, Any]]:
    facts: list[dict[str, Any]] = []
    pattern = re.compile(
        rf"\b(?P<a>{NAME_RE})\s+(?P<verb>has|owns|carries|receives|gets|takes|loses|gives)\s+(?P<object>[^.!?;]+)",
        re.IGNORECASE,
    )
    for match in pattern.finditer(sentence):
        facts.append(
            make_fact(
                "possession",
                clean_name(match.group("a")),
                match.group("verb").lower(),
                metadata,
                object_=trim_value(match.group("object")),
                evidence=sentence,
            )
        )
    return facts


def extract_world_rules(sentence: str, metadata: dict[str, Any]) -> list[dict[str, Any]]:
    lowered = sentence.lower()
    has_rule_language = any(word in lowered for word in ["cannot", "can't", "never", "always", "must", "impossible"])
    has_world_signal = any(word in lowered for word in ["magic", "system", "curse", "law", "rule", "contract", "oath"])
    if not (has_rule_language and has_world_signal):
        return []

    return [
        make_fact(
            "world_rule",
            "World",
            "rule",
            metadata,
            value=trim_value(sentence),
            evidence=sentence,
        )
    ]


def extract_events(sentence: str, metadata: dict[str, Any]) -> list[dict[str, Any]]:
    facts: list[dict[str, Any]] = []
    met_pattern = re.compile(
        rf"\b(?P<a>{NAME_RE})\s+(?:met|meets|first\s+met)\s+(?P<b>{NAME_RE})\b",
        re.IGNORECASE,
    )
    for match in met_pattern.finditer(sentence):
        facts.append(
            make_fact(
                "event",
                clean_name(match.group("a")),
                "met",
                metadata,
                object_=clean_name(match.group("b")),
                evidence=sentence,
                confidence=0.65,
            )
        )
    if facts:
        return facts

    names = [name for name in extract_names(sentence) if name not in STOP_NAMES]
    if len(names) < 2:
        return []

    return [
        make_fact(
            "event",
            names[0],
            "appears_with",
            metadata,
            object_=", ".join(names[1:4]),
            value=trim_value(sentence),
            evidence=sentence,
            confidence=0.45,
        )
    ]


def extract_names(text: str) -> list[str]:
    seen = set()
    names = []
    for match in re.finditer(rf"\b{NAME_RE}\b", text):
        name = clean_name(match.group(0))
        first_word = name.split()[0] if name else ""
        if name in STOP_NAMES or first_word in STOP_NAMES or len(name) < 2:
            continue
        if name.lower() not in seen:
            seen.add(name.lower())
            names.append(name)
    return names
