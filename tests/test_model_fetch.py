import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.providers import extract_gemini_model_names, extract_model_names


def test_extract_ollama_model_names() -> None:
    payload = {
        "models": [
            {"name": "llama3.2:3b", "model": "llama3.2:3b"},
            {"model": "mistral:latest"},
        ]
    }

    assert extract_model_names(payload, keys=("name", "model")) == ["llama3.2:3b", "mistral:latest"]


def test_extract_openai_compatible_model_names() -> None:
    payload = {
        "data": [
            {"id": "gpt-4o-mini"},
            {"id": "gpt-4o"},
        ]
    }

    assert extract_model_names(payload, containers=("data",), keys=("id",)) == ["gpt-4o", "gpt-4o-mini"]


def test_extract_gemini_model_names_filters_generation_models() -> None:
    payload = {
        "models": [
            {"name": "models/gemini-2.5-flash", "supportedGenerationMethods": ["generateContent"]},
            {"name": "models/text-embedding-004", "supportedGenerationMethods": ["embedContent"]},
        ]
    }

    assert extract_gemini_model_names(payload) == ["gemini-2.5-flash"]
