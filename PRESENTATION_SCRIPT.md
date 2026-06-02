# LoreLock Presentation Script

Use `PRESENTATION.html` for the slide deck. Press the right arrow to advance. Press `N` if you want speaker notes on screen, and `F` for fullscreen.

## 30-Second Opening

My project is LoreLock, a story continuity AI agent for fiction writers and editors. The problem is that long-form stories accumulate facts about characters, relationships, secrets, locations, objects, and world rules, and contradictions can slip in during revision. LoreLock accepts story text, extracts candidate continuity facts, lets the writer approve facts into memory, and then checks new scenes against that approved memory. The key design choice is that AI helps extract facts, but deterministic Python rules generate the final continuity warnings.

## Slide Talk Track

### 1. Title

LoreLock is a lightweight AI agent for story continuity. It is built as a Streamlit prototype and focuses on helping writers catch possible contradictions before publication or review.

### 2. Problem

Continuity management becomes difficult when a story gets longer or when scenes are drafted out of order. A writer may remember the main plot but still miss smaller details, like a secret being revealed too early or a place name changing between drafts. LoreLock is aimed at writers, editors, beta readers, writing students, and small story teams.

### 3. Agent Goal

The agent's goal is to protect consistency while keeping the writer in control. It reads story text, extracts candidate facts, asks the user to approve memory, checks a new scene, and reports risks with evidence. That full loop makes it agentic rather than just a one-response chatbot.

### 4. Design Choice

The system separates extraction from judgment. The AI provider proposes structured JSON facts, but those facts are normalized, validated, and approved by the user. The final continuity checks are deterministic Python rules, which makes the warnings more explainable and reduces the risk of hallucinated judgments.

### 5. Architecture

The app is modular. `app.py` launches Streamlit, `src/ui.py` handles the interface, `src/providers.py` handles model providers, `src/extraction.py` extracts facts, `src/facts.py` normalizes and validates them, and `src/continuity.py` checks for conflicts. Memory is stored locally as JSON so the prototype is easy to run and demo.

### 6. What It Detects

LoreLock can flag relationship conflicts, status conflicts, possession conflicts, knowledge leaks, lifecycle risks, world-rule risks, identity drift, setting drift, and contradictions inside the same scene. The output is framed as warnings because some contradictions in fiction are intentional.

### 7. Demo Flow

The demo has four steps. First, I paste source story material and extract candidate facts. Second, I approve useful facts into memory. Third, I paste a conflicting continuation. Fourth, LoreLock reports possible continuity risks with evidence and suggested next actions.

### 8. Implementation

The implementation is intentionally lightweight. It uses Python and Streamlit, with `streamlit>=1.35` as the listed runtime dependency. Provider calls use standard-library HTTP utilities. The app supports Ollama, Gemini, Claude, OpenAI-compatible APIs, and a private heuristic fallback.

### 9. Testing

The project includes automated tests, and the latest verification run had 33 tests passing. The tests cover the review workflow, provider settings, fallback extraction, relationship logic, identity drift, setting drift, same-scene contradictions, and coverage recovery.

### 10. Responsible AI

There are three main responsible AI concerns. First is privacy, because external providers may receive submitted draft text. Second is hallucination, because models can invent or merge facts. Third is overreliance, because not every warning is a real mistake. LoreLock addresses these by making provider routing visible, requiring human approval before memory storage, grounding facts in evidence, and treating warnings as review prompts.

### 11. Conclusion

LoreLock satisfies the project goal by implementing an AI agent that perceives text, reasons through a multi-step workflow, retains approved memory, uses optional provider APIs, applies decision rules, and produces explainable output. Future work would include SQLite memory, a relationship graph view, batch chapter ingestion, stronger alias handling, and vector search for large writing projects.

## Quick Q&A Answers

**What makes this an agent?**

It performs a multi-step goal-oriented workflow: reads text, extracts facts, validates them, stores approved memory, scopes relevant memory, checks a new scene, and reports warnings.

**How do you handle hallucinations?**

LLM output is not trusted automatically. It must be parsed as JSON, normalized, grounded in evidence, and approved by the user before becoming memory. The final warnings come from deterministic rules.

**What happens if the API fails?**

The app can show provider debug information and, if enabled, use a private heuristic fallback. That fallback has lower confidence but keeps the demo usable.

**Why not let the AI rewrite the scene?**

Because contradictions may be intentional. The system is designed as an editorial assistant, so it flags risks and leaves judgment to the writer.

**What is the main limitation?**

Extraction quality. If the model or fallback misses an important fact, the checker may not have enough structured data to flag the issue.
