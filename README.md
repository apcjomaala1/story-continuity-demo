# LoreLock

A generic Streamlit demo for a lightweight story continuity agent.

LoreLock is no longer seeded with a specific story. You paste or upload story material, the app extracts candidate continuity facts, you approve the facts that should become memory, and then the app checks new scenes against that approved memory.

## What It Does

1. Ingests arbitrary story text, outlines, character sheets, or lore notes.
2. Uses a selected LLM provider to extract candidate facts such as relationships, reveals, status changes, possessions, events, world rules, and timeline markers.
3. Lets the writer approve or reject those facts.
4. Stores approved memory locally in the Streamlit session, with optional local JSON save.
5. Checks draft scenes for possible continuity conflicts.

## AI Provider Options

- **Ollama API**: sends selected text to the configured Ollama API endpoint.
- **Gemini API**: sends selected text to the configured Gemini API endpoint.
- **Claude API**: sends selected text to the configured Claude API endpoint.
- **OpenAI API**: sends selected text to the configured OpenAI API endpoint.
- **Private heuristic fallback**: only used if the selected provider fails and fallback is enabled. It is low accuracy and not the intended primary extractor.

The continuity checker itself is deterministic and explainable. The LLM, if enabled, is used for extraction rather than final judgment.

The OpenAI API option uses the OpenAI chat completions request shape. If another provider accepts that same request shape, enter that provider's base URL in the OpenAI API base URL field.

Choose a project folder in the sidebar. LoreLock automatically loads and saves `lorelock_memory.json` and `provider_settings.json` in that folder. The app only keeps the last selected project-folder path in `data/project_settings.json`.

Each provider section can refresh available models from the configured API and show them as a dropdown. If the provider cannot list models, the app falls back to a manual model-name field.

Provider output is passed through a canonicalization gate before it becomes approveable memory. Facts must be atomic, grounded in their evidence snippet, and normalized to controlled predicates; prose-shaped or unsupported claims are filtered instead of treated as checkable facts.

## Extraction Depth

The app asks for an extraction depth, then suggests an excerpt length:

- **Quick**: suggests about 1,500 characters. Best for one short scene beat or a focused contradiction check.
- **Standard**: suggests about 4,000 characters. Best default for one scene or tight partial chapter.
- **Deep**: suggests about 8,000 characters. Best when the scene depends on broader context.

This applies to both memory ingestion and scene checking because both steps perform fact extraction. The suggested length is not a hard limit: the text box counter turns yellow near the target and red when it exceeds it. Longer excerpts give more context, but they also cost more, run slower, and produce more candidate facts to review. Privacy is controlled by the extraction provider choice, not by this setting.

When pasted text exceeds the suggested length, the app can optionally split it into suggested-length parts before extraction. Splitting prefers sentence and newline boundaries and makes one provider/API call per part, so it is off by default.

The main story text boxes use a small local component so character counts update while typing instead of relying on Streamlit's native apply cycle.

Relationship facts are stored as distinct roles on the same subject/object edge. For example, `lover` and `boss` can both apply to the same character pair without being treated as a contradiction, while explicitly conflicting statuses such as `wife` versus `girlfriend` still produce a warning.

## Run

```powershell
pip install -r requirements.txt
streamlit run app.py
```

Then open the local URL printed by Streamlit.

Optional provider environment variables:

```powershell
$env:GEMINI_API_KEY="..."
$env:GEMINI_MODEL="gemini-3.5-flash"
$env:ANTHROPIC_API_KEY="..."
$env:CLAUDE_MODEL="claude-sonnet-4-6"
```

## Project Shape

- `app.py` - Streamlit app, extraction providers, local heuristic extractor, continuity checker
- `components/live_textarea/` - local textarea component with live character counting
- `docs/erd.md` - conceptual ERD for the full continuity memory model
- `docs/demo-plan.md` - presentation-oriented demo plan
- `data/` - optional local memory saves and provider settings; JSON files in this folder are gitignored

## Privacy Notes

`data/*.json`, `exports/*.json`, `.env`, and `.streamlit/secrets.toml` are ignored so story memory, provider settings, exports, and API keys do not accidentally enter git.
