from __future__ import annotations

import json
import os
import re
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

from src.constants import (
    EXCERPT_PROFILES,
    PROVIDER_BASE_URL_KEYS,
    PROVIDER_MODE_ALIASES,
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
    if provider.mode == "Ollama API":
        return f"ollama:{provider.ollama_model}"
    if provider.mode == "Gemini API":
        return f"gemini:{provider.gemini_model}"
    if provider.mode == "Claude API":
        return f"claude:{provider.claude_model}"
    return f"openai:{provider.api_model}"


# ---------------------------------------------------------------------------
# Provider HTTP calls
# ---------------------------------------------------------------------------


def call_ollama(text: str, metadata: dict[str, Any], provider: ProviderConfig) -> str:
    return call_model_for_json(
        extraction_system_prompt(provider),
        extraction_user_prompt(text, metadata, provider),
        provider,
    )


def call_model_for_json(system_prompt: str, user_prompt: str, provider: ProviderConfig) -> str:
    profile = excerpt_profile(provider)
    if provider.mode == "Ollama API":
        body = {
            "model": provider.ollama_model,
            "stream": False,
            "format": "json",
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "options": {
                "temperature": profile["temperature"],
                "num_predict": profile["max_output_tokens"],
            },
        }
        return post_json(f"{provider.ollama_url}/api/chat", body)["message"]["content"]

    if provider.mode == "Gemini API":
        if not provider.gemini_key:
            raise ValueError("Gemini API key is missing")

        body = {
            "contents": [
                {
                    "role": "user",
                    "parts": [
                        {
                            "text": f"{system_prompt}\n\n{user_prompt}"
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

    if provider.mode == "Claude API":
        if not provider.claude_key:
            raise ValueError("Claude API key is missing")

        body = {
            "model": provider.claude_model,
            "max_tokens": profile["max_output_tokens"],
            "temperature": profile["temperature"],
            "system": system_prompt,
            "messages": [
                {
                    "role": "user",
                    "content": user_prompt,
                }
            ],
        }
        headers = {
            "x-api-key": provider.claude_key,
            "anthropic-version": "2023-06-01",
        }
        response = post_json(f"{provider.claude_url}/messages", body, headers)
        return "".join(block.get("text", "") for block in response.get("content", []) if block.get("type") == "text")

    if not provider.api_key:
        raise ValueError("OpenAI API key is missing")

    body = {
        "model": provider.api_model,
        "temperature": profile["temperature"],
        "max_tokens": profile["max_output_tokens"],
        "response_format": {"type": "json_object"},
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    }
    headers = {"Authorization": f"Bearer {provider.api_key}"}
    return post_json(f"{provider.api_url}/chat/completions", body, headers)["choices"][0]["message"]["content"]


def call_gemini(text: str, metadata: dict[str, Any], provider: ProviderConfig) -> str:
    return call_model_for_json(
        extraction_system_prompt(provider),
        extraction_user_prompt(text, metadata, provider),
        provider,
    )


def call_claude(text: str, metadata: dict[str, Any], provider: ProviderConfig) -> str:
    return call_model_for_json(
        extraction_system_prompt(provider),
        extraction_user_prompt(text, metadata, provider),
        provider,
    )


def call_openai_compatible(text: str, metadata: dict[str, Any], provider: ProviderConfig) -> str:
    return call_model_for_json(
        extraction_system_prompt(provider),
        extraction_user_prompt(text, metadata, provider),
        provider,
    )


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
        "Your goal is COMPREHENSIVE coverage: a later scene will be checked against these facts "
        "to find contradictions, so every concrete, verifiable detail matters. "
        "Return only JSON with a top-level key named facts. "
        "Grounding rules: "
        "(1) Extract explicitly stated facts (confidence 1.0). "
        "(2) Also extract strongly implied facts that any attentive reader would infer from the text "
        "(confidence 0.8) — e.g. if the text describes tide cliffs, harbors, and ocean spray, "
        "infer the setting is coastal; if a bell 'breathes' and changes color, infer it is alive/magical; "
        "if someone is called 'Prince' and 'heir to the Storm Seat', infer royal lineage. "
        "(3) Never fabricate details that have no textual basis at all. "
        "Normalize before output: every fact must be one atomic, directly checkable claim. "
        "Use canonical snake_case predicates for the real attribute, relation, action, or rule; "
        "never use wrapper predicates like has_*, have_*, is_*, was_*, or became_*. "
        "Skip claims that cannot be made atomic or grounded. "
        f"Extraction depth: {profile['label']}. {profile['guidance']} "
        f"Return at most {profile['max_facts']} facts."
    )


def extraction_user_prompt(text: str, metadata: dict[str, Any], provider: ProviderConfig) -> str:
    profile = excerpt_profile(provider)
    schema = {
        "facts": [
            {
                "type": "relationship | knowledge | status | possession | location | event | world_rule | trait | ability | timeline",
                "subject": "entity",
                "predicate": "canonical_snake_case",
                "object": "target/place/item/topic/info; empty for simple attributes",
                "value": "short state/attribute/role; empty for simple targets",
                "relation_type": "relationship role only",
                "relation_types": ["optional multiple relationship roles"],
                "known_by": ["explicit knowers only"],
                "evidence": "short grounded snippet",
                "confidence": 0.0,
            }
        ]
    }
    return (
        f"Metadata: {json.dumps(metadata, ensure_ascii=False)}\n\n"
        f"Maximum facts: {profile['max_facts']}\n\n"
        "YOUR TASK: Extract every concrete, verifiable detail from this text that a continuity checker "
        "could use to catch contradictions in a later scene. Be thorough. Err on the side of extracting "
        "more facts rather than fewer. A missing fact means a missed contradiction.\n\n"
        "EXTRACTION CHECKLIST — sweep the text for each category:\n"
        "1. IDENTITY: Full names (including family/house names), titles, aliases, honorifics for every named character.\n"
        "2. APPEARANCE: Hair color/style/length, eye color, skin tone, build, height, distinguishing marks for every described character.\n"
        "3. ROLES & RANKS: Academy roles, candidacies, seat rankings (e.g. 'first seat', 'fifth seat'), "
        "occupations, titles (prince, headmistress, registrar, professor).\n"
        "4. AFFILIATIONS: House membership, faction, unaligned status, scholarship status.\n"
        "5. RELATIONSHIPS: Family ties (parent, sibling), romantic connections, friendships, rivalries, enmities, "
        "mentor/student bonds. Include the emotional quality if stated (tense, warm, hostile).\n"
        "6. POSSESSIONS & ARTIFACTS: Every named object, key, weapon, creature/companion, signet, comb, trunk. "
        "Include material, color, shape, and any magical properties described.\n"
        "7. NAMED LOCATIONS: Every named place (academy name, gates, towers, dormitories, halls, observatories, "
        "orchards, bridges, lakes, libraries, spires). Include physical descriptions and properties "
        "(coastal vs desert, suspended lake, whale-bone arches, folding bridge, etc.).\n"
        "8. SETTING PROPERTIES: Geography (coastal cliffs, desert dunes, ocean), architecture style, "
        "magical atmosphere (living bells, star-water, animated armor).\n"
        "9. WORLD RULES & MAGIC SYSTEMS: Any stated rule, restriction, oath, compact, magical law, "
        "gate behavior, or condition (e.g. 'gate speaks legal names', 'opens at first frost', "
        "'walls bleed on founding anniversaries').\n"
        "10. EVENTS & ACTIONS: Concrete things that happen: arrivals, announcements, threats, "
        "visions, oaths sworn, seals placed, disappearances, deaths (real or faked).\n"
        "11. VERBATIM MESSAGES: Any written warning, prophecy, threat, or inscription — "
        "extract the exact wording as the evidence.\n"
        "12. KNOWLEDGE & SECRETS: Who knows what, who learned what, what was revealed, "
        "what remains hidden. Include the knower explicitly.\n"
        "13. PERSONALITY & BEHAVIOR: Stated personality traits (secretive, cheerful, cold, warm), "
        "habits, characteristic behaviors, attitudes toward others or things.\n"
        "14. TIMELINE & BACKSTORY: Ages, durations ('sixteen years ago'), temporal markers "
        "('first day', 'before dawn'), sequence of past events.\n"
        "15. IMPLIED FACTS (confidence 0.8): Extract facts that are strongly implied but not "
        "explicitly stated. These are critical for continuity checking. Examples:\n"
        "   - Text says 'tide cliffs', 'harbor', 'ocean struck the rocks' → location academy geography value='coastal'.\n"
        "   - Text says the bell 'breathed' and 'changed color' → trait academy_bell nature value='living/magical'.\n"
        "   - Text says 'Prince Caelan, heir to the Storm Seat' → status Caelan title value='prince'; status Caelan lineage value='royal'.\n"
        "   - Text says 'dark-haired like the queen in portraits' → trait Caelan hair_color value='dark'; relationship Caelan mother queen.\n"
        "   - Text says she 'had never stood inside a school of magic before' → status Liora enrollment_status value='new student / first day'.\n"
        "   - Text says 'mainland bridge folded into the water at moonrise' → trait mainland_bridge behavior value='folds into water at moonrise'.\n"
        "   - Text says 'Glassmere hung like a lens of captured sky' → trait Glassmere nature value='suspended lake'.\n"
        "   - Text says someone 'spoke only in gestures' around a character → trait character communication value='uses gestures not speech'.\n\n"
        "CONFIDENCE GUIDE:\n"
        "- 1.0 = directly stated in the text (e.g. 'Mara Venn, scholarship student').\n"
        "- 0.8 = strongly implied, any reader would agree (e.g. coastal setting from ocean/cliffs/harbor descriptions).\n"
        "- 0.6 = probable inference but debatable (use sparingly, only for important continuity details).\n"
        "- Do NOT extract speculative or thematic interpretations.\n\n"
        "NORMALIZATION RULES:\n"
        "- One fact = one claim. Split mixed sentences; skip vague summaries, mood, narration, and unsupported inference.\n"
        "- No type=character. Profile data becomes status, trait, ability, possession, or location.\n"
        "- Strip helper verbs from predicates: has_age -> age, was_student -> role, has_hair_color -> hair_color.\n"
        "- Type map: relationship A->B role; knowledge knower learned/knows info; status age/role/life_status/affiliation; "
        "trait hair_color/height/build/personality; ability ability/skill/limitation; possession owns/carries/lost/gained item; "
        "location current_location/residence/origin; event concrete_action target; world_rule rule/restriction topic; "
        "timeline before/after/during/same_time_as/story_order.\n"
        "- Slot discipline is strict. For status, trait, and ability facts, put the attribute result in value and leave object empty. "
        "Examples: status affiliation value='House Ardent'; status role value='student'; trait hair_color value='black'.\n"
        "- For location facts, put the place in object and leave value empty. Use location origin for birthplace/hometown/origin places, "
        "not status origin. Example: location origin object='Bracken Parish'.\n"
        "- For possession facts, put the item in object and leave value empty. Different owned items are separate facts, not conflicts.\n"
        "- Use object for targets/items/places/topics/info; use value for short states/attributes/roles. "
        "Use both only when the shape needs both, such as relationship or family_business_involvement.\n"
        "- Knowledge subject is the explicit knower/learner, not necessarily the speaker. known_by lists only explicit knowers.\n"
        "- If a sentence names multiple people, attach status, role, trait, ability, and possession facts to the person the phrase "
        "grammatically describes. Do not assign one character's role to the POV character or nearest earlier name.\n\n"
        "Examples:\n"
        "- 'Reika is 26' -> status Reika age = 26.\n"
        "- 'Seraphine belongs to House Ardent' -> status Seraphine affiliation value=House Ardent, object empty.\n"
        "- 'Liora is from Bracken Parish' -> location Liora origin object=Bracken Parish, value empty.\n"
        "- 'Liora owns an iron key' -> possession Liora owns object=iron key, value empty.\n"
        "- 'Sena was a student when she met Kaito and was not involved in the family business' -> "
        "status Sena role=student; event Sena met Kaito; status Sena family_business_involvement family business=not involved.\n"
        "- 'A is B's lover and boss' -> one relationship fact with relation_types ['lover','boss'] or two relationship facts.\n"
        "- 'The Tide Gate is two arches of whale bone' -> trait Tide Gate material value='whale bone'; trait Tide Gate structure value='two arches grown together'.\n"
        "- 'RETURN THE SEALED STAR OR THE PRINCE DROWNS' -> event warning inscription value='RETURN THE SEALED STAR OR THE PRINCE DROWNS'.\n"
        "- 'Dorian was secretive and cold' -> trait Dorian personality value='secretive'; trait Dorian personality value='cold'.\n"
        "- 'Pip is a brass mechanical kestrel' -> trait Pip species value='mechanical kestrel'; trait Pip material value='brass'.\n"
        "- 'Third seat: Caelan Thorne' -> status Caelan Thorne ranking value='third seat'.\n\n"
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
        "mode": "Ollama API",
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
        "retry_missed_with_ai": True,
    }


def load_provider_settings_from_disk(settings_file: Path | None = None) -> dict[str, Any]:
    settings_file = settings_file or PROVIDER_SETTINGS_FILE
    settings = default_provider_settings()
    if not settings_file.exists():
        return settings

    try:
        saved = json.loads(settings_file.read_text(encoding="utf-8"))
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
    settings["mode"] = PROVIDER_MODE_ALIASES.get(settings["mode"], settings["mode"])
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
        "retry_missed_with_ai": provider.retry_missed_with_ai,
    }


def save_provider_settings_to_disk(provider: ProviderConfig, settings_file: Path | None = None) -> None:
    settings_file = settings_file or PROVIDER_SETTINGS_FILE
    settings_file.parent.mkdir(parents=True, exist_ok=True)
    content = json.dumps(provider_settings_payload(provider), indent=2, ensure_ascii=False)
    try:
        if settings_file.exists() and settings_file.read_text(encoding="utf-8") == content:
            return
    except OSError:
        pass
    settings_file.write_text(content, encoding="utf-8")


def ensure_provider_settings_state(settings_file: Path | None = None) -> None:
    import streamlit as st

    settings_file = settings_file or PROVIDER_SETTINGS_FILE
    defaults = default_provider_settings()
    loaded_settings = load_provider_settings_from_disk(settings_file)
    settings_key = str(settings_file.resolve())

    if st.session_state.get("provider_settings_loaded_from") != settings_key:
        for key, value in loaded_settings.items():
            st.session_state[f"provider_{key}"] = value
            st.session_state.pop(f"provider_{key}_widget", None)
            st.session_state.pop(f"provider_{key}_select_widget", None)
        st.session_state.provider_settings_loaded_from = settings_key

    for key, value in loaded_settings.items():
        state_key = f"provider_{key}"
        if state_key not in st.session_state:
            st.session_state[state_key] = value

    for key in PROVIDER_BASE_URL_KEYS:
        state_key = f"provider_{key}"
        if not str(st.session_state.get(state_key, "")).strip():
            st.session_state[state_key] = defaults[key]

    mode = PROVIDER_MODE_ALIASES.get(st.session_state.get("provider_mode"), st.session_state.get("provider_mode"))
    if mode not in PROVIDER_MODES:
        st.session_state.provider_mode = defaults["mode"]
    else:
        st.session_state.provider_mode = mode
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
