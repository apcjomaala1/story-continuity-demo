from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path


APP_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = APP_DIR / "data"
MEMORY_FILE = DATA_DIR / "working_memory.json"
PROVIDER_SETTINGS_FILE = DATA_DIR / "provider_settings.json"
PROVIDER_MODES = [
    "Ollama local LLM",
    "Gemini API",
    "Claude API",
    "OpenAI-compatible API",
]
PROVIDER_BASE_URL_KEYS = {
    "ollama_url",
    "gemini_url",
    "claude_url",
    "api_url",
}

FACT_TYPES = [
    "character",
    "relationship",
    "knowledge",
    "status",
    "possession",
    "location",
    "event",
    "world_rule",
    "trait",
    "ability",
    "timeline",
]

RELATIONSHIP_WORDS = [
    "wife",
    "husband",
    "spouse",
    "partner",
    "lover",
    "girlfriend",
    "boyfriend",
    "ex-girlfriend",
    "ex-boyfriend",
    "ex-wife",
    "ex-husband",
    "mother",
    "father",
    "sister",
    "brother",
    "friend",
    "classmate",
    "ally",
    "enemy",
    "boss",
    "manager",
    "employee",
    "subordinate",
    "mentor",
    "handler",
    "rival",
    "fiance",
    "fiancee",
]

RELATIONSHIP_ACTIONS = [
    "dated",
    "married",
    "betrayed",
    "killed",
    "loves",
    "hates",
    "trusts",
    "distrusts",
    "protects",
    "follows",
]

RELATIONSHIP_ROLE_ALIASES = {
    **{word: word for word in RELATIONSHIP_WORDS},
    **{word: word for word in RELATIONSHIP_ACTIONS},
    "ex girlfriend": "ex-girlfriend",
    "ex-boyfriend": "ex-boyfriend",
    "ex boyfriend": "ex-boyfriend",
    "ex-wife": "ex-wife",
    "ex wife": "ex-wife",
    "ex-husband": "ex-husband",
    "ex husband": "ex-husband",
    "fiancee": "fiance",
}

RELATIONSHIP_ROLE_PATTERN = "|".join(
    re.escape(role).replace(r"\ ", r"[\s_-]+").replace(r"\-", r"[\s_-]+")
    for role in sorted(RELATIONSHIP_ROLE_ALIASES, key=len, reverse=True)
)
RELATIONSHIP_ROLE_LIST_PATTERN = (
    rf"(?:{RELATIONSHIP_ROLE_PATTERN})"
    rf"(?:\s*(?:,|/|&|\band\b|\bor\b|\balso\b|\bas\s+well\s+as\b)\s*(?:{RELATIONSHIP_ROLE_PATTERN}))*"
)
RELATIONSHIP_GENERIC_PREDICATES = {
    "",
    "relationship",
    "relationship_to",
    "relationship-to",
    "relation",
    "relation_to",
    "relation-to",
    "related_to",
    "related-to",
    "is",
    "was",
    "became",
    "becomes",
}
RELATIONSHIP_DIMENSIONS = {
    "wife": "romantic",
    "husband": "romantic",
    "spouse": "romantic",
    "partner": "romantic",
    "lover": "romantic",
    "girlfriend": "romantic",
    "boyfriend": "romantic",
    "ex-girlfriend": "romantic",
    "ex-boyfriend": "romantic",
    "ex-wife": "romantic",
    "ex-husband": "romantic",
    "fiance": "romantic",
    "dated": "romantic",
    "married": "romantic",
    "mother": "family",
    "father": "family",
    "sister": "family",
    "brother": "family",
    "friend": "alliance",
    "classmate": "association",
    "ally": "alliance",
    "enemy": "alliance",
    "rival": "alliance",
    "boss": "authority",
    "manager": "authority",
    "employee": "authority",
    "subordinate": "authority",
    "mentor": "authority",
    "handler": "authority",
    "loves": "affection",
    "hates": "affection",
    "trusts": "trust",
    "distrusts": "trust",
    "protects": "loyalty",
    "follows": "loyalty",
    "betrayed": "loyalty",
    "killed": "harm",
}
RELATIONSHIP_CONFLICT_PAIRS = {
    frozenset(("wife", "girlfriend")),
    frozenset(("husband", "boyfriend")),
    frozenset(("spouse", "girlfriend")),
    frozenset(("spouse", "boyfriend")),
    frozenset(("spouse", "fiance")),
    frozenset(("friend", "enemy")),
    frozenset(("ally", "enemy")),
    frozenset(("loves", "hates")),
    frozenset(("trusts", "distrusts")),
    frozenset(("protects", "betrayed")),
}

NAME_RE = r"[A-Z][A-Za-z]*(?:\s+[A-Z][A-Za-z]*){0,2}"
STOP_NAMES = {
    "A",
    "An",
    "And",
    "As",
    "At",
    "But",
    "By",
    "Chapter",
    "He",
    "Her",
    "His",
    "I",
    "In",
    "It",
    "No",
    "On",
    "She",
    "That",
    "The",
    "Then",
    "They",
    "This",
    "We",
    "When",
}

EXCERPT_PROFILES = {
    "Quick scene slice": {
        "label": "Quick",
        "recommended_chars": 1500,
        "max_facts": 18,
        "temperature": 0.0,
        "max_output_tokens": 1800,
        "guidance": "Best for one short scene beat, a small rewrite, or a focused contradiction check.",
    },
    "Normal scene/chapter excerpt": {
        "label": "Standard",
        "recommended_chars": 4000,
        "max_facts": 36,
        "temperature": 0.1,
        "max_output_tokens": 3200,
        "guidance": "Best default. Usually enough for one scene or a tight partial chapter excerpt.",
    },
    "Deep chapter pass": {
        "label": "Deep",
        "recommended_chars": 8000,
        "max_facts": 72,
        "temperature": 0.1,
        "max_output_tokens": 6000,
        "guidance": "Use when the scene depends on broader context. Slower and more expensive.",
    },
}


@dataclass(frozen=True)
class ProviderConfig:
    mode: str
    extraction_depth: str
    ollama_url: str
    ollama_model: str
    gemini_url: str
    gemini_key: str
    gemini_model: str
    claude_url: str
    claude_key: str
    claude_model: str
    api_url: str
    api_key: str
    api_model: str
    fallback_to_heuristic: bool


@dataclass(frozen=True)
class ContinuityIssue:
    category: str
    severity: str
    message: str
    evidence: str
    suggestion: str
