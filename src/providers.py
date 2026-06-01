from __future__ import annotations

import json
import os
import re
import urllib.error
import urllib.request
from typing import Any

from src.constants import (
    DATA_DIR,
    EXCERPT_PROFILES,
    PROVIDER_BASE_URL_KEYS,
    PROVIDER_MODES,
    PROVIDER_SETTINGS_FILE,
    ProviderConfig,
)
from src.utils import as_text, sorted_unique, summarize_exception, truncate_debug_text


def excerpt_profile(provider: ProviderConfig) -> dict[str, Any]:
    return EXCERPT_PROFILES.get(provider.extraction_depth, EXCERPT_PROFILES["Normal scene/chapter excerpt"])


def suggested_char_limit(provider: ProviderConfig) -> int:
    return int(excerpt_profile(provider)["recommended_chars"])


def provider_source_label(provider: ProviderConfig) -> str:
    if provider.mode == "Ollama local LLM":
        return f"ollama:{provider.ollama_model}"
    if provider.mode == "Gemini API":
        return f"gemini:{provider.gemini_model}"
    if provider.mode == "Claude API":
        return f"claude:{provider.claude_model}"
    return f"api:{provider.api_model}"


# ---------------------------------------------------------------------------
# Provider HTTP calls
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Model fetching
# ---------------------------------------------------------------------------


def fetch_ollama_models(base_url: str) -> list[str]:
    payload = get_json(f"{base_url.rstrip('/')}/api/tags")
    return extract_model_names(payload, keys=("name", "model"))


def fetch_gemini_models(base_url: str, api_key: str) -> list[str]:
    if not api_key:
        raise ValueError("Gemini API key is missing")

    payload = get_json(f"{base_url.rstrip('/')}/models", {"x-goog-api-key": api_key})
    return extract_gemini_model_names(payload)


def fetch_claude_models(base_url: str, api_key: str) -> list[str]:
    if not api_key:
        raise ValueError("Claude API key is missing")

    payload = get_json(
        f"{base_url.rstrip('/')}/models",
        {
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01",
        },
    )
    return extract_model_names(payload, containers=("data", "models"), keys=("id", "name", "model"))


def fetch_openai_compatible_models(base_url: str, api_key: str) -> list[str]:
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
    payload = get_json(f"{base_url.rstrip('/')}/models", headers)
    return extract_model_names(payload, containers=("data", "models"), keys=("id", "name", "model"))


def extract_gemini_model_names(payload: dict[str, Any]) -> list[str]:
    models = []
    for item in payload.get("models", []) if isinstance(payload, dict) else []:
        if not isinstance(item, dict):
            continue
        methods = item.get("supportedGenerationMethods", [])
        if methods and "generateContent" not in methods:
            continue
        name = as_text(item.get("name"))
        if name.startswith("models/"):
            name = name.split("/", 1)[1]
        if name:
            models.append(name)
    return sorted_unique(models)


def extract_model_names(
    payload: dict[str, Any],
    *,
    containers: tuple[str, ...] = ("models", "data"),
    keys: tuple[str, ...] = ("id", "name", "model"),
) -> list[str]:
    if not isinstance(payload, dict):
        return []

    names = []
    for container in containers:
        items = payload.get(container, [])
        if not isinstance(items, list):
            continue
        for item in items:
            if isinstance(item, str):
                names.append(item)
                continue
            if not isinstance(item, dict):
                continue
            for key in keys:
                name = as_text(item.get(key))
                if name:
                    names.append(name)
                    break
    return sorted_unique(names)


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# JSON parsing and debug helpers
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# HTTP helpers
# ---------------------------------------------------------------------------


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


def get_json(url: str, headers: dict[str, str] | None = None) -> dict[str, Any]:
    request = urllib.request.Request(
        url,
        headers={"Accept": "application/json", **(headers or {})},
        method="GET",
    )
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
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


# ---------------------------------------------------------------------------
# Provider settings persistence
# ---------------------------------------------------------------------------


def env_or_default(name: str, default: str) -> str:
    return os.getenv(name) or default


def default_provider_settings() -> dict[str, Any]:
    return {
        "mode": "Ollama local LLM",
        "extraction_depth": "Normal scene/chapter excerpt",
        "ollama_url": "http://localhost:11434",
        "ollama_model": "llama3.2:3b",
        "gemini_url": env_or_default("GEMINI_API_BASE", "https://generativelanguage.googleapis.com/v1beta"),
        "gemini_key": os.getenv("GEMINI_API_KEY", ""),
        "gemini_model": env_or_default("GEMINI_MODEL", "gemini-3.5-flash"),
        "claude_url": env_or_default("ANTHROPIC_API_BASE", "https://api.anthropic.com/v1"),
        "claude_key": os.getenv("ANTHROPIC_API_KEY", ""),
        "claude_model": env_or_default("CLAUDE_MODEL", "claude-sonnet-4-6"),
        "api_url": env_or_default("OPENAI_BASE_URL", "https://api.openai.com/v1"),
        "api_key": os.getenv("OPENAI_API_KEY", ""),
        "api_model": env_or_default("OPENAI_MODEL", "gpt-4o-mini"),
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
            if value.strip() or key not in PROVIDER_BASE_URL_KEYS:
                settings[key] = value

    return normalized_provider_settings(settings)

def normalized_provider_settings(settings: dict[str, Any]) -> dict[str, Any]:
    defaults = default_provider_settings()
    if settings["mode"] not in PROVIDER_MODES:
        settings["mode"] = defaults["mode"]
    if settings["extraction_depth"] not in EXCERPT_PROFILES:
        settings["extraction_depth"] = defaults["extraction_depth"]
    for key in PROVIDER_BASE_URL_KEYS:
        if not str(settings.get(key, "")).strip():
            settings[key] = defaults[key]
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


def ensure_provider_settings_state() -> None:
    import streamlit as st

    defaults = default_provider_settings()

    if not st.session_state.get("provider_settings_initialized"):
        for key, value in load_provider_settings_from_disk().items():
            st.session_state.setdefault(f"provider_{key}", value)
        st.session_state.provider_settings_initialized = True

    for key in PROVIDER_BASE_URL_KEYS:
        state_key = f"provider_{key}"
        if not str(st.session_state.get(state_key, "")).strip():
            st.session_state[state_key] = defaults[key]

    if st.session_state.get("provider_mode") not in PROVIDER_MODES:
        st.session_state.provider_mode = defaults["mode"]
    if st.session_state.get("provider_extraction_depth") not in EXCERPT_PROFILES:
        st.session_state.provider_extraction_depth = defaults["extraction_depth"]


def current_provider_settings() -> dict[str, Any]:
    import streamlit as st

    settings = default_provider_settings()
    for key, default_value in settings.items():
        settings[key] = st.session_state.get(f"provider_{key}", default_value)
        if key in PROVIDER_BASE_URL_KEYS and not str(settings[key]).strip():
            settings[key] = default_value
            st.session_state[f"provider_{key}"] = default_value

    return normalized_provider_settings(settings)
