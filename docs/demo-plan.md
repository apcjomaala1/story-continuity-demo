# Demo Plan

## Goal

Show a generic writing continuity agent that can build memory from arbitrary story material, then check a new scene for contradictions.

## Demo Flow

1. Run `streamlit run app.py`.
2. Use **Ollama API** for a local-provider demo, or use Gemini API, Claude API, or OpenAI API if keys are configured.
3. Use **Standard** extraction depth.
4. Paste a short character sheet, outline, or chapter excerpt into **Ingest Memory**.
5. Click **Extract candidate memory**.
6. Approve the useful facts.
7. Open **Memory Graph** and show the structured memory.
8. Open **Check Scene**, paste a scene with an intentional contradiction, and run the check.
9. Explain that Ollama API, Gemini API, Claude API, or OpenAI API can improve extraction quality when privacy constraints allow it.
10. Explain that the private heuristic is only a low-accuracy fallback if the selected model fails.

## Suggested Generic Test Memory

```text
Mara is Dain's wife.
Dain died in Chapter 4.
Mara learned that Ilya works for the king in Chapter 8.
The silver gate was destroyed in Chapter 5.
Magic cannot heal old injuries.
```

Approve the extracted facts.

## Suggested Contradictory Scene

```text
In Chapter 6, Mara knew that Ilya works for the king.
Dain walked into the room and gave Mara the silver gate key.
Magic healed Mara's old scar completely.
Mara introduced herself as Dain's girlfriend.
```

Expected warnings:

- possible knowledge leak before the Chapter 8 reveal
- lifecycle risk because Dain is dead after Chapter 4
- world rule risk about old injuries and magic
- relationship status conflict if wife/girlfriend facts both exist, while compatible roles like lover/boss on the same pair are allowed

## Agent Architecture

```text
story text
  -> extraction provider
      -> optional Ollama API
      -> optional OpenAI API
      -> optional Gemini API
      -> optional Claude API
      -> private heuristic fallback on failure
  -> candidate facts
  -> human approval
  -> local memory graph
  -> scoped continuity checks
  -> warnings with evidence and suggested next action
```

The fuller data model is documented in `docs/erd.md`. The key architectural idea is that approved memory is not just a flat list of notes: it is a graph of entities, scenes, facts, relationships, knowledge states, evidence spans, and continuity issues.

## Why This Is Lightweight

- It does not ask the model to understand an entire novel at once.
- It stores only approved structured facts.
- It checks only the memory relevant to the current scene.
- It separates in-world story order from narrative chapter order.
- It can run fully private through local Ollama.

## Extraction Depth Strategy

The UI asks for extraction depth, then suggests a matching excerpt length:

- **Quick**: about 1,500 characters for a small focused check.
- **Standard**: about 4,000 characters for the default demo.
- **Deep**: about 8,000 characters when the scene depends on broader context.

This applies to both memory ingestion and scene checking because both steps extract facts. Privacy is controlled by the extraction provider. Excerpt size only controls how much context the selected provider analyzes.

## Next Build Steps

1. Add editable relationship graph views.
2. Add automatic chapter batch ingestion.
3. Add richer `known_by` and `revealed_to` handling.
4. Add a local vector search layer for large projects.
5. Add optional LLM-based explanation after deterministic checks.
