from __future__ import annotations

import json
import os
import re
import uuid
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import streamlit as st
import streamlit.components.v1 as components


APP_DIR = Path(__file__).resolve().parent
DATA_DIR = APP_DIR / "data"
MEMORY_FILE = DATA_DIR / "working_memory.json"
PROVIDER_SETTINGS_FILE = DATA_DIR / "provider_settings.json"
PROVIDER_MODES = [
    "Ollama local LLM",
    "Gemini API",
    "Claude API",
    "OpenAI-compatible API",
]
LIVE_TEXTAREA = components.declare_component(
    "live_textarea",
    path=str(APP_DIR / "components" / "live_textarea"),
)

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


def main() -> None:
    st.set_page_config(page_title="LoreLock", page_icon="LL", layout="wide")
    apply_compact_styles()
    ensure_state()

    st.title("LoreLock")
    st.caption("A generic story continuity agent with private-first memory extraction.")

    provider = render_provider_sidebar()
    tabs = st.tabs(["Ingest Memory", "Memory Graph", "Check Scene", "Privacy"])

    with tabs[0]:
        render_ingest_tab(provider)
    with tabs[1]:
        render_memory_tab()
    with tabs[2]:
        render_check_tab(provider)
    with tabs[3]:
        render_privacy_tab()


def apply_compact_styles() -> None:
    st.markdown(
        """
        <style>
        [data-testid="stTextArea"] textarea {
            font-size: 0.82rem !important;
            line-height: 1.38 !important;
        }

        [data-testid="stTextArea"] label p {
            font-size: 0.86rem !important;
        }

        [data-testid="stAlert"] {
            font-size: 0.86rem !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def ensure_state() -> None:
    st.session_state.setdefault("memory", [])
    st.session_state.setdefault("candidates", [])
    st.session_state.setdefault("scene_facts", [])
    st.session_state.setdefault("issues", [])
    st.session_state.setdefault("provider_notice", "")
    st.session_state.setdefault("provider_debug", "")
    ensure_provider_settings_state()


def default_provider_settings() -> dict[str, Any]:
    return {
        "mode": "Ollama local LLM",
        "extraction_depth": "Normal scene/chapter excerpt",
        "ollama_url": "http://localhost:11434",
        "ollama_model": "llama3.2:3b",
        "gemini_url": os.getenv("GEMINI_API_BASE", "https://generativelanguage.googleapis.com/v1beta"),
        "gemini_key": os.getenv("GEMINI_API_KEY", ""),
        "gemini_model": os.getenv("GEMINI_MODEL", "gemini-3.5-flash"),
        "claude_url": os.getenv("ANTHROPIC_API_BASE", "https://api.anthropic.com/v1"),
        "claude_key": os.getenv("ANTHROPIC_API_KEY", ""),
        "claude_model": os.getenv("CLAUDE_MODEL", "claude-sonnet-4-6"),
        "api_url": os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"),
        "api_key": os.getenv("OPENAI_API_KEY", ""),
        "api_model": os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
        "fallback_to_heuristic": True,
    }


def load_provider_settings_from_disk() -> dict[str, Any]:
    settings = default_provider_settings()
    if not PROVIDER_SETTINGS_FILE.exists():
        return settings

    try:
        saved = json.loads(PROVIDER_SETTINGS_FILE.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return settings

    if not isinstance(saved, dict):
        return settings

    for key, default_value in settings.items():
        value = saved.get(key)
        if isinstance(default_value, bool):
            if isinstance(value, bool):
                settings[key] = value
        elif isinstance(value, str):
            settings[key] = value

    if settings["mode"] not in PROVIDER_MODES:
        settings["mode"] = "Ollama local LLM"
    if settings["extraction_depth"] not in EXCERPT_PROFILES:
        settings["extraction_depth"] = "Normal scene/chapter excerpt"
    return settings


def ensure_provider_settings_state() -> None:
    if st.session_state.get("provider_settings_initialized"):
        return

    for key, value in load_provider_settings_from_disk().items():
        st.session_state.setdefault(f"provider_{key}", value)
    st.session_state.provider_settings_initialized = True


def current_provider_settings() -> dict[str, Any]:
    settings = default_provider_settings()
    for key, default_value in settings.items():
        settings[key] = st.session_state.get(f"provider_{key}", default_value)

    if settings["mode"] not in PROVIDER_MODES:
        settings["mode"] = "Ollama local LLM"
    if settings["extraction_depth"] not in EXCERPT_PROFILES:
        settings["extraction_depth"] = "Normal scene/chapter excerpt"
    return settings


def provider_settings_payload(provider: ProviderConfig) -> dict[str, Any]:
    return {
        "mode": provider.mode,
        "extraction_depth": provider.extraction_depth,
        "ollama_url": provider.ollama_url,
        "ollama_model": provider.ollama_model,
        "gemini_url": provider.gemini_url,
        "gemini_key": provider.gemini_key,
        "gemini_model": provider.gemini_model,
        "claude_url": provider.claude_url,
        "claude_key": provider.claude_key,
        "claude_model": provider.claude_model,
        "api_url": provider.api_url,
        "api_key": provider.api_key,
        "api_model": provider.api_model,
        "fallback_to_heuristic": provider.fallback_to_heuristic,
    }


def save_provider_settings_to_disk(provider: ProviderConfig) -> None:
    DATA_DIR.mkdir(exist_ok=True)
    content = json.dumps(provider_settings_payload(provider), indent=2, ensure_ascii=False)
    try:
        if PROVIDER_SETTINGS_FILE.exists() and PROVIDER_SETTINGS_FILE.read_text(encoding="utf-8") == content:
            return
    except OSError:
        pass
    PROVIDER_SETTINGS_FILE.write_text(content, encoding="utf-8")


def render_provider_sidebar() -> ProviderConfig:
    with st.sidebar:
        st.header("AI Provider")
        mode = st.selectbox(
            "Extraction mode",
            PROVIDER_MODES,
            key="provider_mode",
            help="The provider extracts candidate story facts. Continuity checks remain deterministic.",
        )
        extraction_depth = st.selectbox(
            "Extraction depth",
            list(EXCERPT_PROFILES.keys()),
            key="provider_extraction_depth",
            format_func=lambda key: EXCERPT_PROFILES[key]["label"],
            help="Choose how deep the extractor should look. The app then suggests a matching excerpt length.",
        )
        profile = EXCERPT_PROFILES[extraction_depth]
        st.caption(f"Suggested excerpt: {profile['recommended_chars']:,} chars")
        st.info(profile["guidance"])

        if mode == "Ollama local LLM":
            st.info("Text is sent to a local Ollama server only.")
            ollama_url = st.text_input("Ollama URL", key="provider_ollama_url")
            ollama_model = st.text_input("Ollama model", key="provider_ollama_model")
        elif mode == "Gemini API":
            st.warning("Gemini mode sends selected story text to Google's Gemini API.")
            gemini_url = st.text_input("Gemini API base URL", key="provider_gemini_url")
            gemini_model = st.text_input("Gemini model", key="provider_gemini_model")
            gemini_key = st.text_input("Gemini API key", type="password", key="provider_gemini_key")
        elif mode == "Claude API":
            st.warning("Claude mode sends selected story text to Anthropic's Claude API.")
            claude_url = st.text_input("Claude API base URL", key="provider_claude_url")
            claude_model = st.text_input("Claude model", key="provider_claude_model")
            claude_key = st.text_input("Claude API key", type="password", key="provider_claude_key")
        else:
            st.warning("API mode sends selected story text to an external service.")
            api_url = st.text_input("API base URL", key="provider_api_url")
            api_model = st.text_input("Model", key="provider_api_model")
            api_key = st.text_input("API key", type="password", key="provider_api_key")

        provider_values = current_provider_settings()
        ollama_url = provider_values["ollama_url"]
        ollama_model = provider_values["ollama_model"]
        gemini_url = provider_values["gemini_url"]
        gemini_key = provider_values["gemini_key"]
        gemini_model = provider_values["gemini_model"]
        claude_url = provider_values["claude_url"]
        claude_key = provider_values["claude_key"]
        claude_model = provider_values["claude_model"]
        api_url = provider_values["api_url"]
        api_key = provider_values["api_key"]
        api_model = provider_values["api_model"]

        st.caption(
            f"Provider settings are saved locally to `{PROVIDER_SETTINGS_FILE.name}`. "
            "Privacy is controlled by the extraction provider."
        )
        fallback_to_heuristic = st.checkbox(
            "Use private heuristic fallback if extraction provider fails",
            key="provider_fallback_to_heuristic",
            help="The heuristic fallback is low accuracy. It exists only so the demo can still run when a model is unavailable.",
        )

        st.divider()
        st.metric("Approved facts", len(st.session_state.memory))
        if st.button("Load saved memory", use_container_width=True):
            load_memory_from_disk()
        if st.button("Save memory locally", use_container_width=True):
            save_memory_to_disk()

    provider = ProviderConfig(
        mode=mode,
        extraction_depth=extraction_depth,
        ollama_url=ollama_url.rstrip("/"),
        ollama_model=ollama_model.strip(),
        gemini_url=gemini_url.rstrip("/"),
        gemini_key=gemini_key.strip(),
        gemini_model=gemini_model.strip(),
        claude_url=claude_url.rstrip("/"),
        claude_key=claude_key.strip(),
        claude_model=claude_model.strip(),
        api_url=api_url.rstrip("/"),
        api_key=api_key.strip(),
        api_model=api_model.strip(),
        fallback_to_heuristic=fallback_to_heuristic,
    )
    save_provider_settings_to_disk(provider)
    return provider


def excerpt_profile(provider: ProviderConfig) -> dict[str, Any]:
    return EXCERPT_PROFILES.get(provider.extraction_depth, EXCERPT_PROFILES["Normal scene/chapter excerpt"])


def suggested_char_limit(provider: ProviderConfig) -> int:
    return int(excerpt_profile(provider)["recommended_chars"])


def render_live_textarea(
    label: str,
    *,
    value: str,
    placeholder: str,
    key: str,
    provider: ProviderConfig,
    seed_token: str = "",
    height: int = 360,
) -> str:
    profile = excerpt_profile(provider)
    state_key = f"{key}_value"
    current_value = st.session_state.get(state_key, value)
    if seed_token and seed_token != st.session_state.get(f"{key}_seed_token"):
        current_value = value
        st.session_state[state_key] = value
        st.session_state[f"{key}_seed_token"] = seed_token

    result = LIVE_TEXTAREA(
        label=label,
        value=current_value,
        placeholder=placeholder,
        height=height,
        suggested_chars=int(profile["recommended_chars"]),
        seed_token=seed_token,
        default=current_value,
        key=key,
    )
    if isinstance(result, str):
        st.session_state[state_key] = result
        return result
    return current_value


def render_chunking_option(key: str, text: str, provider: ProviderConfig) -> bool:
    limit = suggested_char_limit(provider)
    if len(text) <= limit:
        return False

    chunks = split_text_into_chunks(text, limit)
    sizes = ", ".join(f"{len(chunk):,}" for chunk in chunks)
    st.warning(
        f"This text is {len(text):,} chars; the current suggestion is {limit:,}. "
        f"Optional splitting would make {len(chunks)} extraction calls."
    )
    enabled = st.checkbox(
        "Split into suggested-length parts before extraction",
        value=False,
        key=f"{key}_split_over_limit",
        help=(
            "Keeps each part under the suggested character count and prefers newline or sentence boundaries. "
            "This increases provider/API calls and cost."
        ),
    )
    if enabled:
        st.caption(f"Planned chunk sizes: {sizes} chars.")
    return enabled


def render_ingest_tab(provider: ProviderConfig) -> None:
    st.subheader("Build Story Memory")
    st.write(
        "Paste a chapter, scene, outline, character sheet, or lore note. "
        "The agent extracts candidate continuity facts, then you approve what becomes canon."
    )

    meta_col, text_col = st.columns([0.34, 0.66], gap="large")
    with meta_col:
        metadata = render_metadata_form("ingest", default_source="story note")
        uploaded = st.file_uploader("Optional text file", type=["txt", "md"])

    uploaded_text = ""
    seed_token = ""
    if uploaded is not None:
        uploaded_text = uploaded.getvalue().decode("utf-8", errors="replace")
        metadata["source"] = uploaded.name
        seed_token = f"{uploaded.name}:{len(uploaded_text)}"

    with text_col:
        text = render_live_textarea(
            "Source text",
            value=uploaded_text,
            placeholder="Paste story material here.",
            key="source_text_live",
            provider=provider,
            seed_token=seed_token,
        )
        split_over_limit = render_chunking_option("source_text_live", text, provider)

    if st.button("Extract candidate memory", type="primary", use_container_width=True):
        if not text.strip():
            st.error("Paste or upload story text first.")
        else:
            with st.spinner("Extracting candidate facts..."):
                facts, notice, debug = extract_facts_for_text(text, metadata, provider, split_over_limit=split_over_limit)
            st.session_state.candidates = facts
            st.session_state.provider_notice = notice
            st.session_state.provider_debug = debug

    render_provider_feedback("ingest")

    render_candidate_facts()


def render_metadata_form(prefix: str, default_source: str) -> dict[str, Any]:
    source = st.text_input("Source label", value=default_source, key=f"{prefix}_source")
    chapter = st.number_input("Narrative chapter", min_value=0, max_value=9999, value=1, key=f"{prefix}_chapter")
    story_order = st.number_input(
        "Story order index",
        min_value=0,
        max_value=999999,
        value=int(chapter),
        key=f"{prefix}_story_order",
        help="Use this for flashbacks or nonlinear stories. Earlier in-world events should have lower numbers.",
    )
    scene_time = st.text_input("Story time", value="", placeholder="Day 4, three years later, winter, etc.", key=f"{prefix}_time")
    location = st.text_input("Location", value="", key=f"{prefix}_location")
    pov = st.text_input("POV character", value="", key=f"{prefix}_pov")
    return {
        "source": source.strip() or default_source,
        "chapter": int(chapter),
        "story_order": int(story_order),
        "scene_time": scene_time.strip(),
        "location": location.strip(),
        "pov": pov.strip(),
    }


def render_candidate_facts() -> None:
    candidates = st.session_state.candidates
    if not candidates:
        st.info("No candidate facts extracted yet.")
        return

    st.subheader(f"Canonical Candidate Facts: {len(candidates)}")
    with st.form("approve_candidates"):
        approved_ids = []
        for fact in candidates:
            label = fact_label(fact)
            checked = st.checkbox(label, value=True, key=f"approve_{fact['id']}")
            with st.expander("Evidence and metadata", expanded=False):
                st.json(fact)
            if checked:
                approved_ids.append(fact["id"])

        submitted = st.form_submit_button("Add approved facts to memory", type="primary")

    if submitted:
        existing_ids = {fact["id"] for fact in st.session_state.memory}
        selected = [
            fact
            for fact in candidates
            if fact["id"] in approved_ids and fact["id"] not in existing_ids
        ]
        st.session_state.memory.extend(selected)
        st.success(f"Added {len(selected)} fact(s) to memory.")


def render_memory_tab() -> None:
    st.subheader("Approved Story Memory")
    memory = st.session_state.memory

    if not memory:
        st.info("Memory is empty. Ingest story text first or import JSON below.")
    else:
        rows = [
            {
                "id": fact["id"],
                "type": fact["type"],
                "subject": fact["subject"],
                "predicate": fact["predicate"],
                "object": fact["object"],
                "value": fact["value"],
                "relation_type": fact.get("relation_type", ""),
                "relation_dimension": fact.get("relation_dimension", ""),
                "chapter": fact["chapter"],
                "story_order": fact["story_order"],
                "source": fact["source"],
            }
            for fact in memory
        ]
        st.dataframe(rows, hide_index=True, use_container_width=True)

        labels = [fact_label(fact) for fact in memory]
        selected = st.multiselect("Facts to remove", labels)
        if st.button("Remove selected facts"):
            st.session_state.memory = [
                fact
                for fact in memory
                if fact_label(fact) not in selected
            ]
            st.rerun()

    st.divider()
    st.download_button(
        "Download memory JSON",
        data=json.dumps(memory, indent=2, ensure_ascii=False),
        file_name="lorelock_memory.json",
        mime="application/json",
        use_container_width=True,
    )

    imported = st.file_uploader("Import memory JSON", type=["json"], key="memory_import")
    if imported is not None and st.button("Load imported memory"):
        data = json.loads(imported.getvalue().decode("utf-8"))
        st.session_state.memory = normalize_facts(data if isinstance(data, list) else data.get("facts", []), {})
        st.success(f"Loaded {len(st.session_state.memory)} fact(s).")

    raw_json = st.text_area(
        "Manual JSON editor",
        value=json.dumps(memory, indent=2, ensure_ascii=False),
        height=260,
    )
    if st.button("Replace memory from editor"):
        try:
            data = json.loads(raw_json)
            st.session_state.memory = normalize_facts(data if isinstance(data, list) else data.get("facts", []), {})
            st.success("Memory replaced.")
        except json.JSONDecodeError as exc:
            st.error(f"Invalid JSON: {exc}")


def render_check_tab(provider: ProviderConfig) -> None:
    st.subheader("Check a New Scene")
    st.write(
        "Paste a draft scene. The agent extracts temporary scene facts, scopes relevant memory, "
        "and flags possible continuity issues."
    )

    meta_col, text_col = st.columns([0.34, 0.66], gap="large")
    with meta_col:
        metadata = render_metadata_form("check", default_source="draft scene")

    with text_col:
        scene_text = render_live_textarea(
            "Draft scene",
            placeholder="Paste the scene you want to check.",
            value="",
            key="scene_text_live",
            provider=provider,
        )
        split_over_limit = render_chunking_option("scene_text_live", scene_text, provider)

    if st.button("Extract and check scene", type="primary", use_container_width=True):
        if not scene_text.strip():
            st.error("Paste a scene first.")
        elif not st.session_state.memory:
            st.warning("Memory is empty. You can still extract scene facts, but there is nothing to compare against.")
            facts, notice, debug = extract_facts_for_text(scene_text, metadata, provider, split_over_limit=split_over_limit)
            st.session_state.scene_facts = facts
            st.session_state.issues = []
            st.session_state.provider_notice = notice
            st.session_state.provider_debug = debug
        else:
            with st.spinner("Extracting scene facts and checking continuity..."):
                facts, notice, debug = extract_facts_for_text(scene_text, metadata, provider, split_over_limit=split_over_limit)
                issues = check_continuity(scene_text, facts, st.session_state.memory)
            st.session_state.scene_facts = facts
            st.session_state.issues = issues
            st.session_state.provider_notice = notice
            st.session_state.provider_debug = debug

    render_provider_feedback("scene")

    col_a, col_b = st.columns([0.48, 0.52], gap="large")
    with col_a:
        st.markdown("#### Extracted Scene Facts")
        if st.session_state.scene_facts:
            for fact in st.session_state.scene_facts:
                with st.container(border=True):
                    st.markdown(f"**{fact_label(fact)}**")
                    st.caption(f"Evidence: {fact.get('evidence', '')}")
        else:
            st.info("No scene facts extracted yet.")

    with col_b:
        st.markdown(f"#### Continuity Warnings: {len(st.session_state.issues)}")
        if st.session_state.issues:
            severity_rank = {"high": 0, "medium": 1, "low": 2}
            for issue in sorted(st.session_state.issues, key=lambda item: severity_rank[item.severity]):
                with st.container(border=True):
                    st.markdown(f"**{issue.category}** - `{issue.severity.upper()}`")
                    st.write(issue.message)
                    st.caption(f"Evidence: {issue.evidence}")
                    st.caption(f"Next: {issue.suggestion}")
        else:
            st.info("No warnings yet.")


def render_privacy_tab() -> None:
    st.subheader("Privacy Model")
    st.write(
        "LoreLock is designed so the story memory and continuity rules can run locally. "
        "Fact extraction uses the selected model provider. The low-accuracy heuristic is only a failure fallback."
    )

    st.markdown(
        """
- **Ollama local LLM**: sends selected text to `localhost`. Good privacy if the model is local, slower on weak hardware.
- **Gemini API**: sends selected text to Google's Gemini API.
- **Claude API**: sends selected text to Anthropic's Claude API.
- **OpenAI-compatible API**: sends selected text to an external model endpoint. Better extraction, weaker privacy.
- **Private heuristic fallback**: only used when the selected model fails and fallback is enabled. Low accuracy.
- **Extraction depth**: controls the suggested excerpt length and how many facts the extractor is asked to return.
- **Optional splitting**: over-limit pasted text can be split into suggested-length parts, which makes one extraction call per part.
- **Approved memory only**: extracted facts are suggestions until the writer approves them.
- **Local saves**: saved memory goes to `data/working_memory.json`, which is gitignored.
- **Provider settings**: saved provider choices, endpoints, models, and API keys go to `data/provider_settings.json`, which is gitignored.
"""
    )


def render_provider_feedback(key_prefix: str) -> None:
    notice = st.session_state.get("provider_notice", "")
    debug = st.session_state.get("provider_debug", "")
    if notice:
        if debug:
            st.warning(notice)
        else:
            st.caption(notice)
    if debug:
        with st.expander("Provider debug dump", expanded=True):
            st.text_area(
                "Copy this diagnostic",
                value=debug,
                height=280,
                key=f"{key_prefix}_provider_debug_dump",
            )


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


def provider_source_label(provider: ProviderConfig) -> str:
    if provider.mode == "Ollama local LLM":
        return f"ollama:{provider.ollama_model}"
    if provider.mode == "Gemini API":
        return f"gemini:{provider.gemini_model}"
    if provider.mode == "Claude API":
        return f"claude:{provider.claude_model}"
    return f"api:{provider.api_model}"


def summarize_exception(exc: Exception) -> str:
    message = str(exc).strip()
    if len(message) > 180:
        message = message[:177].rstrip() + "..."
    return f"{type(exc).__name__}: {message}"


def build_provider_debug(
    exc: Exception,
    provider: ProviderConfig,
    metadata: dict[str, Any],
    text: str,
    raw: str,
) -> str:
    profile = excerpt_profile(provider)
    lines = [
        "Provider debug dump",
        f"provider_mode: {provider.mode}",
        f"provider_model: {provider_source_label(provider)}",
        f"extraction_depth: {profile['label']}",
        f"input_chars: {len(text)}",
        f"metadata: {json.dumps(metadata, ensure_ascii=False)}",
        f"exception_type: {type(exc).__name__}",
        f"exception_message: {str(exc)}",
    ]

    if isinstance(exc, json.JSONDecodeError):
        lines.extend(
            [
                f"json_error_line: {exc.lineno}",
                f"json_error_column: {exc.colno}",
                f"json_error_position: {exc.pos}",
                "json_error_context:",
                json_error_context(exc.doc, exc.pos),
            ]
        )

    if raw:
        lines.extend(
            [
                f"raw_response_chars: {len(raw)}",
                "raw_response:",
                truncate_debug_text(raw, 12000),
            ]
        )
    else:
        lines.append("raw_response: <none captured; provider call failed before model text was returned>")

    return "\n".join(lines)


def json_error_context(document: str, position: int, radius: int = 400) -> str:
    start = max(0, position - radius)
    end = min(len(document), position + radius)
    excerpt = document[start:end]
    pointer = " " * max(0, position - start) + "^"
    return f"{excerpt}\n{pointer}"


def truncate_debug_text(value: str, limit: int) -> str:
    if len(value) <= limit:
        return value
    omitted = len(value) - limit
    return f"{value[:limit]}\n\n... <truncated {omitted} chars>"


def call_ollama(text: str, metadata: dict[str, Any], provider: ProviderConfig) -> str:
    profile = excerpt_profile(provider)
    body = {
        "model": provider.ollama_model,
        "stream": False,
        "format": "json",
        "messages": [
            {"role": "system", "content": extraction_system_prompt(provider)},
            {"role": "user", "content": extraction_user_prompt(text, metadata, provider)},
        ],
        "options": {
            "temperature": profile["temperature"],
            "num_predict": profile["max_output_tokens"],
        },
    }
    return post_json(f"{provider.ollama_url}/api/chat", body)["message"]["content"]


def call_gemini(text: str, metadata: dict[str, Any], provider: ProviderConfig) -> str:
    if not provider.gemini_key:
        raise ValueError("Gemini API key is missing")

    profile = excerpt_profile(provider)
    body = {
        "contents": [
            {
                "role": "user",
                "parts": [
                    {
                        "text": (
                            f"{extraction_system_prompt(provider)}\n\n"
                            f"{extraction_user_prompt(text, metadata, provider)}"
                        )
                    }
                ],
            }
        ],
        "generationConfig": {
            "temperature": profile["temperature"],
            "maxOutputTokens": profile["max_output_tokens"],
            "responseMimeType": "application/json",
        },
    }
    url = f"{provider.gemini_url}/models/{provider.gemini_model}:generateContent"
    response = post_json(url, body, {"x-goog-api-key": provider.gemini_key})
    return response["candidates"][0]["content"]["parts"][0]["text"]


def call_claude(text: str, metadata: dict[str, Any], provider: ProviderConfig) -> str:
    if not provider.claude_key:
        raise ValueError("Claude API key is missing")

    profile = excerpt_profile(provider)
    body = {
        "model": provider.claude_model,
        "max_tokens": profile["max_output_tokens"],
        "temperature": profile["temperature"],
        "system": extraction_system_prompt(provider),
        "messages": [
            {
                "role": "user",
                "content": extraction_user_prompt(text, metadata, provider),
            }
        ],
    }
    headers = {
        "x-api-key": provider.claude_key,
        "anthropic-version": "2023-06-01",
    }
    response = post_json(f"{provider.claude_url}/messages", body, headers)
    return "".join(block.get("text", "") for block in response.get("content", []) if block.get("type") == "text")


def call_openai_compatible(text: str, metadata: dict[str, Any], provider: ProviderConfig) -> str:
    if not provider.api_key:
        raise ValueError("API key is missing")

    profile = excerpt_profile(provider)
    body = {
        "model": provider.api_model,
        "temperature": profile["temperature"],
        "max_tokens": profile["max_output_tokens"],
        "response_format": {"type": "json_object"},
        "messages": [
            {"role": "system", "content": extraction_system_prompt(provider)},
            {"role": "user", "content": extraction_user_prompt(text, metadata, provider)},
        ],
    }
    headers = {"Authorization": f"Bearer {provider.api_key}"}
    return post_json(f"{provider.api_url}/chat/completions", body, headers)["choices"][0]["message"]["content"]


def post_json(url: str, body: dict[str, Any], headers: dict[str, str] | None = None) -> dict[str, Any]:
    request = urllib.request.Request(
        url,
        data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json", **(headers or {})},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=90) as response:
            response_text = response.read().decode("utf-8", errors="replace")
            try:
                return json.loads(response_text)
            except json.JSONDecodeError as exc:
                details = truncate_debug_text(response_text.strip(), 4000) if response_text else "<empty response body>"
                raise RuntimeError(f"Provider returned non-JSON HTTP response: {details}") from exc
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        details = truncate_debug_text(body.strip(), 4000) if body else "<empty response body>"
        raise RuntimeError(f"HTTP {exc.code} {exc.reason}: {details}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(str(exc)) from exc


def extraction_system_prompt(provider: ProviderConfig) -> str:
    profile = excerpt_profile(provider)
    return (
        "You extract structured continuity facts for fiction editing. "
        "Return only JSON with a top-level key named facts. "
        "Do not invent facts. Keep each fact grounded in the provided text. "
        "Every fact must be atomic and directly checkable: one subject, one predicate, "
        "and one object or value. Split compound claims into separate facts. "
        "Use short evidence snippets. Prefer facts that matter for continuity: "
        "relationships, status changes, knowledge/reveals, possessions, locations, "
        "world rules, timeline markers, abilities, and traits. "
        f"Extraction depth: {profile['label']}. {profile['guidance']} "
        f"Return at most {profile['max_facts']} facts."
    )


def extraction_user_prompt(text: str, metadata: dict[str, Any], provider: ProviderConfig) -> str:
    profile = excerpt_profile(provider)
    schema = {
        "facts": [
            {
                "type": "relationship | knowledge | status | possession | location | event | world_rule | trait | ability | timeline | character",
                "subject": "entity name",
                "predicate": "short relation or action",
                "object": "entity or target, optional",
                "value": "status/value, optional",
                "relation_type": "for relationship facts only, e.g. lover, boss, friend, enemy",
                "relation_types": ["optional list when one subject-object pair has multiple simultaneous relationship roles"],
                "known_by": ["optional character names"],
                "evidence": "short quote or close paraphrase",
                "confidence": 0.0,
            }
        ]
    }
    return (
        f"Metadata: {json.dumps(metadata, ensure_ascii=False)}\n\n"
        f"Extraction depth: {profile['label']}.\n"
        f"Extraction guidance: {profile['guidance']}\n"
        f"Maximum facts: {profile['max_facts']}\n\n"
        "Atomicity rules:\n"
        "- One fact must express one checkable claim only.\n"
        "- Do not put extra claims in value. Values should be short states like 'student', 'dead', 'not involved'.\n"
        "- Use timeline only for explicit ordering facts such as 'A happens before B'. Do not use timeline for mixed biography/status claims.\n"
        "- Example: 'Sena was a student when she met Kaito and was not involved in the family business' should become separate facts: "
        "status Sena role=student; event Sena met Kaito; status Sena family_business_involvement family business=not involved.\n\n"
        "For multidimensional relationships, keep each role distinct. If A is B's lover and boss, "
        "emit either two relationship facts for the same subject/object or one fact with relation_types "
        "['lover', 'boss']; do not collapse that into one vague relationship value.\n\n"
        f"Return JSON matching this shape:\n{json.dumps(schema, indent=2)}\n\n"
        f"Text:\n{text}"
    )


def parse_json_object(raw: str) -> dict[str, Any]:
    cleaned = raw.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?", "", cleaned).strip()
        cleaned = re.sub(r"```$", "", cleaned).strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start == -1 or end == -1:
            raise
        return json.loads(cleaned[start : end + 1])


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


def canonical_relationship_type(value: Any) -> str:
    text = as_text(value).lower().strip()
    text = re.sub(r"[\s_]+", " ", text)
    text = text.strip(" ,.;:!?\"'()[]")
    if not text:
        return ""
    return RELATIONSHIP_ROLE_ALIASES.get(text, text.replace(" ", "-"))


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
        "confidence": max(0.0, min(1.0, float(confidence))),
        "extraction": "heuristic",
    }


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


def canonicalize_fact(fact: dict[str, Any]) -> dict[str, Any] | None:
    fact = normalize_canonical_slots(fact)
    if has_uncheckable_shape(fact):
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


def canonical_predicate(value: Any) -> str:
    text = as_text(value).lower().strip()
    text = re.sub(r"[\s-]+", "_", text)
    text = re.sub(r"[^a-z0-9_]", "", text)
    return text.strip("_")


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


def relationship_relation_type(fact: dict[str, Any]) -> str:
    terms = relationship_terms_from_fact(fact)
    if terms:
        return terms[0]
    return canonical_relationship_type(fact.get("relation_type", ""))


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


def clean_known_by(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [clean_name(as_text(item)) for item in value if clean_name(as_text(item))]


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


def text_overlap(a: str, b: str) -> bool:
    a_words = significant_words(a.lower())
    b_words = significant_words(b.lower())
    return len(a_words.intersection(b_words)) >= 2


def significant_words(text: str) -> set[str]:
    stop = {"the", "and", "that", "with", "from", "this", "there", "their", "about", "into", "after", "before"}
    return {word for word in re.findall(r"[a-zA-Z]{4,}", text.lower()) if word not in stop}


def fact_label(fact: dict[str, Any]) -> str:
    if fact.get("type") == "relationship":
        relation_type = relationship_relation_type(fact)
        parts = ["relationship", fact.get("subject", "")]
        if relation_type:
            parts.append(relation_type)
        if fact.get("object"):
            parts.append(fact["object"])
        value = fact.get("value", "")
        if value and canonical_relationship_type(value) != relation_type:
            parts.append(f"= {value}")
        return " | ".join(part for part in parts if part)

    parts = [fact.get("type", "fact"), fact.get("subject", "")]
    if fact.get("predicate"):
        parts.append(fact["predicate"])
    if fact.get("object"):
        parts.append(fact["object"])
    if fact.get("value"):
        parts.append(f"= {fact['value']}")
    return " | ".join(part for part in parts if part)


def clean_name(value: str) -> str:
    cleaned = re.sub(r"\s+", " ", value).strip(" ,.;:!?\"'()[]")
    return cleaned


def trim_value(value: str, limit: int = 180) -> str:
    cleaned = re.sub(r"\s+", " ", value).strip(" ,.;:!?\"'()[]")
    if len(cleaned) <= limit:
        return cleaned
    return cleaned[: limit - 3].rstrip() + "..."


def save_memory_to_disk() -> None:
    DATA_DIR.mkdir(exist_ok=True)
    MEMORY_FILE.write_text(
        json.dumps(st.session_state.memory, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    st.sidebar.success(f"Saved to {MEMORY_FILE.name}.")


def load_memory_from_disk() -> None:
    if not MEMORY_FILE.exists():
        st.sidebar.warning("No saved memory file yet.")
        return
    data = json.loads(MEMORY_FILE.read_text(encoding="utf-8"))
    st.session_state.memory = normalize_facts(data if isinstance(data, list) else data.get("facts", []), {})
    st.sidebar.success(f"Loaded {len(st.session_state.memory)} fact(s).")


if __name__ == "__main__":
    main()
