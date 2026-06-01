from __future__ import annotations

import json
from typing import Any

import streamlit as st
import streamlit.components.v1 as components

from src.constants import (
    APP_DIR,
    DATA_DIR,
    EXCERPT_PROFILES,
    MEMORY_FILE,
    PROVIDER_BASE_URL_KEYS,
    PROVIDER_MODES,
    ProviderConfig,
)
from src.continuity import check_continuity
from src.extraction import extract_facts_for_text
from src.facts import normalize_facts
from src.providers import (
    current_provider_settings,
    default_provider_settings,
    ensure_provider_settings_state,
    excerpt_profile,
    fetch_claude_models,
    fetch_gemini_models,
    fetch_ollama_models,
    fetch_openai_compatible_models,
    save_provider_settings_to_disk,
    suggested_char_limit,
)
from src.utils import as_text, fact_label, summarize_exception

LIVE_TEXTAREA = components.declare_component(
    "live_textarea",
    path=str(APP_DIR / "components" / "live_textarea"),
)


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


def render_model_selector(
    label: str,
    *,
    setting_key: str,
    cache_key: str,
    fetch_models: Any,
) -> None:
    options_key = f"{cache_key}_model_options"
    notice_key = f"{cache_key}_model_notice"

    if st.button("Refresh model list", key=f"{cache_key}_refresh_models", use_container_width=True):
        try:
            options = fetch_models()
            if not options:
                raise RuntimeError("No models were returned by the provider.")
        except Exception as exc:
            st.session_state[options_key] = []
            st.session_state[notice_key] = f"Model list unavailable: {summarize_exception(exc)}"
        else:
            st.session_state[options_key] = options
            st.session_state[notice_key] = f"Fetched {len(options)} model(s)."
            if not st.session_state.get(setting_key):
                st.session_state[setting_key] = options[0]

    options = list(st.session_state.get(options_key, []))
    current = as_text(st.session_state.get(setting_key, ""))
    if options:
        choices = options if current in options or not current else [current, *options]
        index = choices.index(current) if current in choices else 0
        selected = st.selectbox(label, choices, index=index, key=f"{setting_key}_select")
        st.session_state[setting_key] = selected
    else:
        st.text_input(label, key=setting_key)

    notice = st.session_state.get(notice_key, "")
    if notice:
        if notice.startswith("Model list unavailable"):
            st.warning(notice)
        else:
            st.caption(notice)


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
            render_model_selector(
                "Ollama model",
                setting_key="provider_ollama_model",
                cache_key="ollama",
                fetch_models=lambda: fetch_ollama_models(ollama_url),
            )
        elif mode == "Gemini API":
            st.warning("Gemini mode sends selected story text to Google's Gemini API.")
            gemini_url = st.text_input("Gemini API base URL", key="provider_gemini_url")
            gemini_key = st.text_input("Gemini API key", type="password", key="provider_gemini_key")
            render_model_selector(
                "Gemini model",
                setting_key="provider_gemini_model",
                cache_key="gemini",
                fetch_models=lambda: fetch_gemini_models(gemini_url, gemini_key),
            )
        elif mode == "Claude API":
            st.warning("Claude mode sends selected story text to Anthropic's Claude API.")
            claude_url = st.text_input("Claude API base URL", key="provider_claude_url")
            claude_key = st.text_input("Claude API key", type="password", key="provider_claude_key")
            render_model_selector(
                "Claude model",
                setting_key="provider_claude_model",
                cache_key="claude",
                fetch_models=lambda: fetch_claude_models(claude_url, claude_key),
            )
        else:
            st.warning("API mode sends selected story text to an external service.")
            api_url = st.text_input("API base URL", key="provider_api_url")
            api_key = st.text_input("API key", type="password", key="provider_api_key")
            render_model_selector(
                "Model",
                setting_key="provider_api_model",
                cache_key="api",
                fetch_models=lambda: fetch_openai_compatible_models(api_url, api_key),
            )

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

        from src.constants import PROVIDER_SETTINGS_FILE as _settings_file

        st.caption(
            f"Provider settings are saved locally to `{_settings_file.name}`. "
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


# ---------------------------------------------------------------------------
# Live textarea and chunking helpers
# ---------------------------------------------------------------------------


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

    from src.extraction import split_text_into_chunks

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


# ---------------------------------------------------------------------------
# Tab renderers
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Memory persistence
# ---------------------------------------------------------------------------


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
