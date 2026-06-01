from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import streamlit as st
import streamlit.components.v1 as components

from src.constants import (
    APP_DIR,
    DATA_DIR,
    EXCERPT_PROFILES,
    FACT_TYPES,
    PROVIDER_MODES,
    PROJECT_SETTINGS_FILE,
    ContinuityIssue,
    ProviderConfig,
)
from src.continuity import check_continuity
from src.extraction import extract_facts_for_text
from src.extraction import infer_scene_metadata
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
    st.caption("A quiet continuity desk for scenes, chapters, character notes, and lore.")

    provider = render_provider_sidebar()
    tabs = st.tabs(["Scene Review", "Memory Graph", "Privacy"])

    with tabs[0]:
        render_scene_review_tab(provider)
    with tabs[1]:
        render_memory_tab()
    with tabs[2]:
        render_privacy_tab()

    autosave_project_state()


def apply_compact_styles() -> None:
    st.markdown(
        """
        <style>
        h1, h2, h3 {
            letter-spacing: 0;
        }

        h1 {
            font-size: 2.45rem !important;
            margin-bottom: 0.1rem !important;
        }

        h3 {
            margin-top: 0.4rem !important;
        }

        .stTabs [data-baseweb="tab-list"] {
            gap: 0.35rem;
        }

        .stTabs [data-baseweb="tab"] {
            border-radius: 8px 8px 0 0;
            font-weight: 650;
        }

        [data-testid="stTextArea"] textarea {
            border-radius: 8px;
            font-size: 0.95rem !important;
            line-height: 1.55 !important;
        }

        [data-testid="stTextArea"] label p {
            font-size: 0.9rem !important;
            font-weight: 650;
        }

        [data-testid="stAlert"] {
            border-radius: 8px;
            font-size: 0.9rem !important;
        }

        [data-testid="stMetric"] {
            border-radius: 8px;
            padding: 0.7rem 0.8rem;
        }

        .stButton button,
        .stDownloadButton button,
        [data-testid="stFormSubmitButton"] button {
            border-radius: 8px;
            font-weight: 650;
        }

        [data-baseweb="input"] input,
        [data-baseweb="select"] > div,
        [data-baseweb="textarea"] textarea {
            border-radius: 8px;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def ensure_state() -> None:
    st.session_state.setdefault("project_folder_input", load_last_project_folder())
    st.session_state.setdefault("memory", [])
    st.session_state.setdefault("candidates", [])
    st.session_state.setdefault("scene_facts", [])
    st.session_state.setdefault("issues", [])
    st.session_state.setdefault("provider_notice", "")
    st.session_state.setdefault("provider_debug", "")
    st.session_state.setdefault("scene_metadata", {})
    st.session_state.setdefault("metadata_notice", "")
    st.session_state.setdefault("metadata_debug", "")
    st.session_state.project_folder_path = str(resolve_project_folder(st.session_state.project_folder_input))


def load_last_project_folder() -> str:
    try:
        data = json.loads(PROJECT_SETTINGS_FILE.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return str(DATA_DIR)
    if isinstance(data, dict) and as_text(data.get("project_folder")).strip():
        return as_text(data["project_folder"]).strip()
    return str(DATA_DIR)


def resolve_project_folder(value: str) -> Path:
    raw = as_text(value).strip() or str(DATA_DIR)
    path = Path(raw).expanduser()
    if not path.is_absolute():
        path = APP_DIR / path
    return path.resolve(strict=False)


def current_project_folder() -> Path:
    return resolve_project_folder(st.session_state.get("project_folder_input", str(DATA_DIR)))


def project_memory_file(project_folder: Path | None = None) -> Path:
    return (project_folder or current_project_folder()) / "lorelock_memory.json"


def project_provider_settings_file(project_folder: Path | None = None) -> Path:
    return (project_folder or current_project_folder()) / "provider_settings.json"


def ensure_project_loaded(project_folder: Path) -> None:
    project_key = str(project_folder)
    if st.session_state.get("project_loaded_from") == project_key:
        return

    try:
        project_folder.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        st.session_state.project_notice = f"Could not open project folder: {summarize_exception(exc)}"
        return

    memory_file = project_memory_file(project_folder)
    if memory_file.exists():
        try:
            data = json.loads(memory_file.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            st.session_state.memory = []
            st.session_state.project_notice = f"Could not load project memory: {summarize_exception(exc)}"
        else:
            st.session_state.memory = normalize_facts(data if isinstance(data, list) else data.get("facts", []), {})
            st.session_state.project_notice = f"Loaded {len(st.session_state.memory)} fact(s) from {memory_file.name}."
    else:
        st.session_state.memory = []
        st.session_state.project_notice = f"Started a new project memory file: {memory_file.name}."

    st.session_state.candidates = []
    st.session_state.scene_facts = []
    st.session_state.issues = []
    st.session_state.scene_metadata = {}
    st.session_state.project_loaded_from = project_key
    st.session_state.project_memory_saved_signature = ""


def autosave_project_state() -> None:
    project_folder = current_project_folder()
    try:
        project_folder.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        st.sidebar.warning(f"Could not open project folder: {summarize_exception(exc)}")
        return

    PROJECT_SETTINGS_FILE.parent.mkdir(parents=True, exist_ok=True)
    project_config = json.dumps({"project_folder": str(project_folder)}, indent=2, ensure_ascii=False)
    try:
        if not PROJECT_SETTINGS_FILE.exists() or PROJECT_SETTINGS_FILE.read_text(encoding="utf-8") != project_config:
            PROJECT_SETTINGS_FILE.write_text(project_config, encoding="utf-8")
    except OSError as exc:
        st.sidebar.warning(f"Could not remember project folder: {summarize_exception(exc)}")

    memory_file = project_memory_file(project_folder)
    content = json.dumps(st.session_state.memory, indent=2, ensure_ascii=False)
    signature = f"{memory_file}:{content}"
    if st.session_state.get("project_memory_saved_signature") == signature:
        return

    try:
        if not memory_file.exists() or memory_file.read_text(encoding="utf-8") != content:
            memory_file.write_text(content, encoding="utf-8")
    except OSError as exc:
        st.sidebar.warning(f"Could not auto-save memory: {summarize_exception(exc)}")
    else:
        st.session_state.project_memory_saved_signature = signature


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
        st.header("Project")
        project_folder_text = st.text_input(
            "Project folder",
            key="project_folder_input",
            help="LoreLock auto-loads and auto-saves JSON files in this folder. Relative paths are resolved from the app folder.",
        )
        project_folder = resolve_project_folder(project_folder_text)
        st.caption(f"Using `{project_folder}`")
        ensure_project_loaded(project_folder)
        ensure_provider_settings_state(project_provider_settings_file(project_folder))
        if st.session_state.get("project_notice"):
            st.caption(st.session_state.project_notice)

        st.divider()
        st.header("Writing Assistant")
        mode = st.selectbox(
            "Reading engine",
            PROVIDER_MODES,
            key="provider_mode",
            help="This reads the selected story text and suggests facts. Continuity checks still run inside LoreLock.",
        )
        extraction_depth = st.selectbox(
            "How closely should it read?",
            list(EXCERPT_PROFILES.keys()),
            key="provider_extraction_depth",
            format_func=lambda key: EXCERPT_PROFILES[key]["label"],
            help="Choose how much detail the reader should look for. LoreLock suggests a matching excerpt length.",
        )
        profile = EXCERPT_PROFILES[extraction_depth]
        st.caption(f"Suggested excerpt: {profile['recommended_chars']:,} chars")
        st.info(profile["guidance"])

        if mode == "Ollama API":
            st.info("Selected text is sent to the configured Ollama API endpoint.")
            ollama_url = st.text_input("Ollama API base URL", key="provider_ollama_url")
            render_model_selector(
                "Ollama model",
                setting_key="provider_ollama_model",
                cache_key="ollama",
                fetch_models=lambda: fetch_ollama_models(ollama_url),
            )
        elif mode == "Gemini API":
            st.info("Selected text is sent to the configured Gemini API endpoint.")
            gemini_url = st.text_input("Gemini API base URL", key="provider_gemini_url")
            gemini_key = st.text_input("Gemini API key", type="password", key="provider_gemini_key")
            render_model_selector(
                "Gemini model",
                setting_key="provider_gemini_model",
                cache_key="gemini",
                fetch_models=lambda: fetch_gemini_models(gemini_url, gemini_key),
            )
        elif mode == "Claude API":
            st.info("Selected text is sent to the configured Claude API endpoint.")
            claude_url = st.text_input("Claude API base URL", key="provider_claude_url")
            claude_key = st.text_input("Claude API key", type="password", key="provider_claude_key")
            render_model_selector(
                "Claude model",
                setting_key="provider_claude_model",
                cache_key="claude",
                fetch_models=lambda: fetch_claude_models(claude_url, claude_key),
            )
        else:
            st.info("Selected text is sent to the configured OpenAI API endpoint.")
            api_url = st.text_input("OpenAI API base URL", key="provider_api_url")
            api_key = st.text_input("OpenAI API key", type="password", key="provider_api_key")
            render_model_selector(
                "OpenAI model",
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

        st.caption(
            f"Provider settings auto-save to `{project_provider_settings_file(project_folder).name}`. "
            "Text routing is controlled by the selected reading engine."
        )
        fallback_to_heuristic = st.checkbox(
            "Use private backup reader if the selected reader fails",
            key="provider_fallback_to_heuristic",
            help="This backup is less accurate. It exists so the demo can still run when a model is unavailable.",
        )

        st.divider()
        st.metric("Story facts saved", len(st.session_state.memory))
        st.caption(f"Story memory auto-saves to `{project_memory_file(project_folder).name}`.")
        if st.button("Reload project memory", use_container_width=True):
            load_memory_from_disk()
        if st.button("Save project memory now", use_container_width=True):
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
    try:
        save_provider_settings_to_disk(provider, project_provider_settings_file(project_folder))
    except OSError as exc:
        st.sidebar.warning(f"Could not auto-save provider settings: {summarize_exception(exc)}")
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


def render_scene_review_tab(provider: ProviderConfig) -> None:
    st.subheader("Review a Scene")
    st.write(
        "Paste a chapter, scene, outline, character sheet, or lore note. "
        "LoreLock pulls out possible story facts, checks them against what you have approved, "
        "and lets you decide what becomes canon."
    )

    uploaded = st.file_uploader("Optional draft file", type=["txt", "md"])
    uploaded_text = ""
    seed_token = ""
    default_source = "draft scene"
    if uploaded is not None:
        uploaded_text = uploaded.getvalue().decode("utf-8", errors="replace")
        default_source = uploaded.name
        seed_token = f"{uploaded.name}:{len(uploaded_text)}"

    text = render_live_textarea(
        "Scene or source text",
        value=uploaded_text,
        placeholder="Paste the scene, chapter, outline, character sheet, or lore note you want checked.",
        key="review_text_live",
        provider=provider,
        seed_token=seed_token,
    )
    split_over_limit = render_chunking_option("review_text_live", text, provider)

    with st.expander("Scene details and optional overrides", expanded=False):
        render_scene_metadata_summary(st.session_state.scene_metadata)
        overrides = render_metadata_form("review", default_source=default_source)

    if st.button("Review this text", type="primary", use_container_width=True):
        if not text.strip():
            st.error("Paste or upload story text first.")
        else:
            with st.spinner("Reading the text, finding scene details, and checking continuity..."):
                facts, issues, notice, debug, metadata = review_text_against_memory(
                    text,
                    overrides,
                    default_source,
                    provider,
                    st.session_state.memory,
                    split_over_limit=split_over_limit,
                )
            st.session_state.candidates = facts
            st.session_state.scene_facts = facts
            st.session_state.issues = issues
            st.session_state.provider_notice = notice
            st.session_state.provider_debug = debug
            st.session_state.scene_metadata = metadata

    render_provider_feedback("review")

    with st.expander(f"Continuity notes ({len(st.session_state.issues)})", expanded=bool(st.session_state.issues)):
        render_continuity_warnings(show_header=False)
    with st.expander(f"Possible story facts ({len(st.session_state.candidates)})", expanded=bool(st.session_state.candidates)):
        render_candidate_facts(show_header=False)


def review_text_against_memory(
    text: str,
    overrides: dict[str, Any],
    default_source: str,
    provider: ProviderConfig,
    memory: list[dict[str, Any]],
    *,
    split_over_limit: bool,
) -> tuple[list[dict[str, Any]], list[ContinuityIssue], str, str, dict[str, Any]]:
    inferred_metadata, metadata_notice, metadata_debug = infer_scene_metadata(text, provider)
    metadata = merge_scene_metadata(default_source, inferred_metadata, overrides)
    facts, notice, debug = extract_facts_for_text(text, metadata, provider, split_over_limit=split_over_limit)
    issues = check_continuity(text, facts, memory) if memory else []
    combined_notice = " ".join(part for part in [metadata_notice, notice] if part)
    combined_debug = "\n\n".join(part for part in [metadata_debug, debug] if part)
    return facts, issues, combined_notice, combined_debug, metadata


def render_metadata_form(prefix: str, default_source: str) -> dict[str, Any]:
    st.caption("Leave these blank and LoreLock will infer what it can from the pasted text.")
    source = st.text_input("Source label", value=default_source, key=f"{prefix}_source")
    chapter = st.text_input("Chapter override", value="", placeholder="Auto", key=f"{prefix}_chapter")
    story_order = st.text_input(
        "Timeline order override",
        value="",
        placeholder="Auto",
        key=f"{prefix}_story_order",
        help="Use this for flashbacks or nonlinear stories. Earlier in-world events should have lower numbers.",
    )
    scene_time = st.text_input("Story time override", value="", placeholder="Auto", key=f"{prefix}_time")
    location = st.text_input("Location", value="", key=f"{prefix}_location")
    pov = st.text_input("POV character", value="", key=f"{prefix}_pov")
    return {
        "source": source.strip(),
        "chapter": chapter.strip(),
        "story_order": story_order.strip(),
        "scene_time": scene_time.strip(),
        "location": location.strip(),
        "pov": pov.strip(),
    }


def merge_scene_metadata(default_source: str, inferred: dict[str, Any], overrides: dict[str, Any]) -> dict[str, Any]:
    chapter = first_int(overrides.get("chapter"), inferred.get("chapter"), 1)
    story_order = first_int(overrides.get("story_order"), inferred.get("story_order"), chapter)
    return {
        "source": first_text(overrides.get("source"), inferred.get("source"), default_source),
        "chapter": chapter,
        "story_order": story_order,
        "scene_time": first_text(overrides.get("scene_time"), inferred.get("scene_time"), ""),
        "location": first_text(overrides.get("location"), inferred.get("location"), ""),
        "pov": first_text(overrides.get("pov"), inferred.get("pov"), ""),
    }


def first_text(*values: Any) -> str:
    for value in values:
        text = as_text(value).strip()
        if text:
            return text
    return ""


def first_int(*values: Any) -> int:
    for value in values:
        try:
            if value not in ("", None):
                return int(value)
        except (TypeError, ValueError):
            continue
    return 0


def render_scene_metadata_summary(metadata: dict[str, Any]) -> None:
    if not metadata:
        st.info("Scene details will be inferred when you review the text.")
        return

    labels = {
        "source": "Source",
        "chapter": "Chapter",
        "story_order": "Timeline order",
        "scene_time": "Story time",
        "location": "Location",
        "pov": "POV",
    }
    for key, label in labels.items():
        value = metadata.get(key)
        if value not in ("", None):
            st.caption(f"{label}: {value}")


def render_candidate_facts(*, show_header: bool = True) -> None:
    candidates = st.session_state.candidates
    if not candidates:
        st.info("No candidate facts extracted yet.")
        return

    if show_header:
        st.subheader(f"Possible story facts: {len(candidates)}")
    st.caption("Edit these suggestions before adding them. Delete rows you do not want, or uncheck Add.")

    rows = [fact_to_editor_row(fact, include_add=True) for fact in candidates]
    edited_rows = st.data_editor(
        rows,
        hide_index=True,
        num_rows="dynamic",
        use_container_width=True,
        key="candidate_fact_editor",
        column_config=fact_editor_columns(include_add=True),
        disabled=["id", "label"],
    )
    edited_facts = editor_rows_to_facts(edited_rows, st.session_state.scene_metadata)
    st.session_state.candidates = edited_facts

    if st.button("Add selected facts to story memory", type="primary", use_container_width=True):
        existing_ids = {fact["id"] for fact in st.session_state.memory}
        edited_by_id = {fact["id"]: fact for fact in edited_facts}
        selected = [
            fact
            for row in editor_rows(edited_rows)
            if row.get("add") and (fact := edited_by_id.get(as_text(row.get("id")))) and fact["id"] not in existing_ids
        ]
        st.session_state.memory.extend(selected)
        st.success(f"Added {len(selected)} fact(s) to story memory.")


def render_memory_tab() -> None:
    st.subheader("Story Memory")
    memory = st.session_state.memory

    if not memory:
        st.info("Story memory is empty. Review story text first or import JSON below.")
    else:
        st.caption("Edit cells directly. Delete a row from the table to remove that fact from memory.")
        edited_rows = st.data_editor(
            [fact_to_editor_row(fact) for fact in memory],
            hide_index=True,
            num_rows="dynamic",
            use_container_width=True,
            key="memory_fact_editor",
            column_config=fact_editor_columns(),
            disabled=["id", "label"],
        )
        edited_memory = editor_rows_to_facts(edited_rows, {})
        if facts_signature(edited_memory) != facts_signature(st.session_state.memory):
            st.session_state.memory = edited_memory
            memory = st.session_state.memory

    st.divider()
    st.download_button(
        "Download story memory JSON",
        data=json.dumps(memory, indent=2, ensure_ascii=False),
        file_name="lorelock_memory.json",
        mime="application/json",
        use_container_width=True,
    )

    imported = st.file_uploader("Import story memory JSON", type=["json"], key="memory_import")
    if imported is not None and st.button("Load imported story memory"):
        data = json.loads(imported.getvalue().decode("utf-8"))
        st.session_state.memory = normalize_facts(data if isinstance(data, list) else data.get("facts", []), {})
        st.success(f"Loaded {len(st.session_state.memory)} fact(s).")

    raw_json = st.text_area(
        "Advanced: edit memory JSON directly",
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


def fact_to_editor_row(fact: dict[str, Any], *, include_add: bool = False) -> dict[str, Any]:
    row = {
        "label": fact_label(fact),
        "id": fact.get("id", ""),
        "type": fact.get("type", ""),
        "subject": fact.get("subject", ""),
        "predicate": fact.get("predicate", ""),
        "object": fact.get("object", ""),
        "value": fact.get("value", ""),
        "known_by": ", ".join(as_text(item) for item in fact.get("known_by", [])),
        "chapter": fact.get("chapter", 0),
        "story_order": fact.get("story_order", 0),
        "scene_time": fact.get("scene_time", ""),
        "location": fact.get("location", ""),
        "pov": fact.get("pov", ""),
        "source": fact.get("source", ""),
        "evidence": fact.get("evidence", ""),
        "confidence": fact.get("confidence", 0.7),
        "extraction": fact.get("extraction", ""),
        "relation_type": fact.get("relation_type", ""),
        "relation_dimension": fact.get("relation_dimension", ""),
    }
    if include_add:
        row = {"add": True, **row}
    return row


def fact_editor_columns(*, include_add: bool = False) -> dict[str, Any]:
    columns: dict[str, Any] = {
        "label": st.column_config.TextColumn("Fact", width="large", help="Readable summary; generated from the editable fields."),
        "id": st.column_config.TextColumn("ID", width="small"),
        "type": st.column_config.SelectboxColumn("Type", options=FACT_TYPES),
        "subject": st.column_config.TextColumn("Subject", width="medium"),
        "predicate": st.column_config.TextColumn("Predicate", width="medium"),
        "object": st.column_config.TextColumn("Object", width="medium"),
        "value": st.column_config.TextColumn("Value", width="medium"),
        "known_by": st.column_config.TextColumn("Known by", width="medium"),
        "chapter": st.column_config.NumberColumn("Chapter", min_value=0, step=1),
        "story_order": st.column_config.NumberColumn("Timeline", min_value=0, step=1),
        "scene_time": st.column_config.TextColumn("Story time", width="medium"),
        "location": st.column_config.TextColumn("Location", width="medium"),
        "pov": st.column_config.TextColumn("POV", width="medium"),
        "source": st.column_config.TextColumn("Source", width="medium"),
        "evidence": st.column_config.TextColumn("Evidence", width="large"),
        "confidence": st.column_config.NumberColumn("Confidence", min_value=0.0, max_value=1.0, step=0.05),
        "extraction": st.column_config.TextColumn("Reader", width="small"),
        "relation_type": st.column_config.TextColumn("Relation", width="small"),
        "relation_dimension": st.column_config.TextColumn("Relation kind", width="small"),
    }
    if include_add:
        return {"add": st.column_config.CheckboxColumn("Add", default=True), **columns}
    return columns


def editor_rows(value: Any) -> list[dict[str, Any]]:
    if hasattr(value, "to_dict"):
        return value.to_dict("records")
    if isinstance(value, list):
        return [row for row in value if isinstance(row, dict)]
    return []


def editor_rows_to_facts(value: Any, metadata: dict[str, Any]) -> list[dict[str, Any]]:
    raw_facts = []
    for row in editor_rows(value):
        subject = editor_text(row.get("subject"))
        if not subject:
            continue
        raw_facts.append(
            {
                "id": editor_text(row.get("id")),
                "type": editor_text(row.get("type")),
                "subject": subject,
                "predicate": editor_text(row.get("predicate")),
                "object": editor_text(row.get("object")),
                "value": editor_text(row.get("value")),
                "known_by": comma_list(row.get("known_by")),
                "chapter": row.get("chapter"),
                "story_order": row.get("story_order"),
                "scene_time": editor_text(row.get("scene_time")),
                "location": editor_text(row.get("location")),
                "pov": editor_text(row.get("pov")),
                "source": editor_text(row.get("source")),
                "evidence": editor_text(row.get("evidence")),
                "confidence": row.get("confidence"),
                "extraction": editor_text(row.get("extraction")) or "edited",
                "relation_type": editor_text(row.get("relation_type")),
                "relation_dimension": editor_text(row.get("relation_dimension")),
            }
        )
    return normalize_facts(raw_facts, metadata, extraction_source="edited")


def editor_text(value: Any) -> str:
    try:
        if value != value:
            return ""
    except TypeError:
        pass
    return as_text(value).strip()


def comma_list(value: Any) -> list[str]:
    if isinstance(value, list):
        return [editor_text(item) for item in value if editor_text(item)]
    return [item.strip() for item in editor_text(value).split(",") if item.strip()]


def facts_signature(facts: list[dict[str, Any]]) -> list[tuple[Any, ...]]:
    return [
        (
            fact.get("id"),
            fact.get("type"),
            fact.get("subject"),
            fact.get("predicate"),
            fact.get("object"),
            fact.get("value"),
            tuple(fact.get("known_by", [])),
            fact.get("chapter"),
            fact.get("story_order"),
            fact.get("scene_time"),
            fact.get("location"),
            fact.get("pov"),
            fact.get("source"),
            fact.get("evidence"),
            fact.get("confidence"),
            fact.get("extraction"),
            fact.get("relation_type", ""),
            fact.get("relation_dimension", ""),
        )
        for fact in facts
    ]


def render_continuity_warnings(*, show_header: bool = True) -> None:
    if show_header:
        st.markdown(f"#### Continuity notes: {len(st.session_state.issues)}")
    if st.session_state.issues:
        severity_rank = {"high": 0, "medium": 1, "low": 2}
        for issue in sorted(st.session_state.issues, key=lambda item: severity_rank[item.severity]):
            with st.container(border=True):
                st.markdown(f"**{issue.category}** - `{issue.severity.upper()}`")
                st.write(issue.message)
                st.caption(f"Where LoreLock noticed it: {issue.evidence}")
                st.caption(f"Possible next step: {issue.suggestion}")
    elif st.session_state.scene_facts and not st.session_state.memory:
        st.info("No approved story memory yet, so this pass only found possible facts.")
    else:
        st.info("No warnings yet.")


def render_privacy_tab() -> None:
    st.subheader("Privacy")
    st.write(
        "LoreLock keeps your approved story memory and continuity checks local. "
        "The selected reading engine receives the text you ask it to review so it can suggest facts."
    )

    st.markdown(
        """
- **Ollama API**: sends selected text to the configured Ollama API endpoint.
- **Gemini API**: sends selected text to the configured Gemini API endpoint.
- **Claude API**: sends selected text to the configured Claude API endpoint.
- **OpenAI API**: sends selected text to the configured OpenAI API endpoint.
- **Private backup reader**: runs locally and is only used when the selected model fails and fallback is enabled.
- **Reading depth**: controls the suggested excerpt length and how many facts the reader is asked to return.
- **Optional splitting**: over-limit pasted text can be split into suggested-length parts, which makes one extraction call per part.
- **Approved memory only**: extracted facts are suggestions until the writer approves them.
- **Project folder**: memory auto-saves to `lorelock_memory.json` in the selected project folder.
- **Provider settings**: provider choices, endpoints, models, and API keys auto-save to `provider_settings.json` in the selected project folder.
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
    memory_file = project_memory_file()
    memory_file.parent.mkdir(parents=True, exist_ok=True)
    memory_file.write_text(
        json.dumps(st.session_state.memory, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    st.session_state.project_memory_saved_signature = f"{memory_file}:{json.dumps(st.session_state.memory, indent=2, ensure_ascii=False)}"
    st.sidebar.success(f"Saved to {memory_file.name}.")


def load_memory_from_disk() -> None:
    memory_file = project_memory_file()
    if not memory_file.exists():
        st.sidebar.warning("No saved memory file yet.")
        return
    data = json.loads(memory_file.read_text(encoding="utf-8"))
    st.session_state.memory = normalize_facts(data if isinstance(data, list) else data.get("facts", []), {})
    st.session_state.project_memory_saved_signature = ""
    st.sidebar.success(f"Loaded {len(st.session_state.memory)} fact(s).")
