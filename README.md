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

- **Ollama local LLM**: sends selected text to a local Ollama server, such as `llama3.2:3b`. Better extraction while keeping drafts local, but may be slow on weak hardware.
- **Gemini API**: sends selected text to Google's Gemini API. Useful for stronger extraction when privacy constraints allow it.
- **Claude API**: sends selected text to Anthropic's Claude API. Useful for stronger extraction when privacy constraints allow it.
- **OpenAI-compatible API**: sends selected text to an external API endpoint. Better extraction, weaker privacy. Only use this deliberately.
- **Private heuristic fallback**: only used if the selected provider fails and fallback is enabled. It is low accuracy and not the intended primary extractor.

The continuity checker itself is deterministic and explainable. The LLM, if enabled, is used for extraction rather than final judgment.

## Extraction Depth

The app asks for an extraction depth, then suggests an excerpt length:

- **Quick**: suggests about 1,500 characters. Best for one short scene beat or a focused contradiction check.
- **Standard**: suggests about 4,000 characters. Best default for one scene or tight partial chapter.
- **Deep**: suggests about 8,000 characters. Best when the scene depends on broader context.

This applies to both memory ingestion and scene checking because both steps perform fact extraction. The suggested length is not a hard limit: the text box counter turns yellow near the target and red when it exceeds it. Longer excerpts give more context, but they also cost more, run slower, and produce more candidate facts to review. Privacy is controlled by the extraction provider choice, not by this setting.

The main story text boxes use a small local component so character counts update while typing instead of relying on Streamlit's native apply cycle.

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
- `data/` - optional local memory saves; JSON files in this folder are gitignored

## Privacy Notes

`data/*.json`, `exports/*.json`, `.env`, and `.streamlit/secrets.toml` are ignored so story memory, exports, and API keys do not accidentally enter git.
