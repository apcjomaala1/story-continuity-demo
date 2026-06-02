import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import providers


def test_default_provider_urls_are_nonempty_when_env_is_blank(monkeypatch) -> None:
    monkeypatch.setenv("GEMINI_API_BASE", "")
    monkeypatch.setenv("ANTHROPIC_API_BASE", "")
    monkeypatch.setenv("OPENAI_BASE_URL", "")

    settings = providers.default_provider_settings()

    assert settings["ollama_url"] == "http://localhost:11434"
    assert settings["gemini_url"] == "https://generativelanguage.googleapis.com/v1beta"
    assert settings["claude_url"] == "https://api.anthropic.com/v1"
    assert settings["api_url"] == "https://api.openai.com/v1"
    assert settings["retry_missed_with_ai"] is False


def test_blank_saved_provider_urls_are_backfilled(tmp_path, monkeypatch) -> None:
    settings_file = tmp_path / "provider_settings.json"
    settings_file.write_text(
        json.dumps(
            {
                "mode": "Gemini API",
                "extraction_depth": "Normal scene/chapter excerpt",
                "ollama_url": "",
                "gemini_url": "",
                "claude_url": "",
                "api_url": "",
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(providers, "PROVIDER_SETTINGS_FILE", settings_file)

    settings = providers.load_provider_settings_from_disk()

    assert settings["ollama_url"] == "http://localhost:11434"
    assert settings["gemini_url"] == "https://generativelanguage.googleapis.com/v1beta"
    assert settings["claude_url"] == "https://api.anthropic.com/v1"
    assert settings["api_url"] == "https://api.openai.com/v1"


def test_retry_missed_with_ai_setting_is_loaded(tmp_path, monkeypatch) -> None:
    settings_file = tmp_path / "provider_settings.json"
    settings_file.write_text(
        json.dumps(
            {
                "retry_missed_with_ai": True,
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(providers, "PROVIDER_SETTINGS_FILE", settings_file)

    settings = providers.load_provider_settings_from_disk()

    assert settings["retry_missed_with_ai"] is True
