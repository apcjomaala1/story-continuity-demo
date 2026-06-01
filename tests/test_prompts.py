import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.constants import ProviderConfig
from src.providers import extraction_user_prompt


def make_provider() -> ProviderConfig:
    return ProviderConfig(
        mode="Ollama API",
        extraction_depth="Normal scene/chapter excerpt",
        ollama_url="http://localhost:11434",
        ollama_model="llama3.2:3b",
        gemini_url="https://generativelanguage.googleapis.com/v1beta",
        gemini_key="",
        gemini_model="gemini-2.5-flash",
        claude_url="https://api.anthropic.com/v1",
        claude_key="",
        claude_model="claude-3-5-sonnet-latest",
        api_url="https://api.openai.com/v1",
        api_key="",
        api_model="gpt-4o-mini",
        fallback_to_heuristic=True,
    )


def test_extraction_prompt_pins_status_location_and_possession_slots() -> None:
    prompt = extraction_user_prompt("Liora is from Bracken Parish.", {"source": "test"}, make_provider())

    assert "For status, trait, and ability facts, put the attribute result in value and leave object empty" in prompt
    assert "Use location origin for birthplace/hometown/origin places, not status origin" in prompt
    assert "For possession facts, put the item in object and leave value empty" in prompt
