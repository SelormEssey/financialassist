"""Tests for defaults, .env loading, and environment overrides."""

from pathlib import Path

import pytest

from app.config import Settings


@pytest.fixture(autouse=True)
def isolate_settings(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Keep local configuration out of the settings tests."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("FINANCIALASSIST_APP_NAME", raising=False)
    monkeypatch.delenv("FINANCIALASSIST_DEBUG", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("FINANCIALASSIST_EMBEDDING_MODEL", raising=False)
    monkeypatch.delenv("FINANCIALASSIST_AGENT_MODEL", raising=False)


def test_settings_use_defaults() -> None:
    settings = Settings()

    assert settings.app_name == "financialassist"
    assert settings.debug is False


def test_settings_load_dotenv(tmp_path: Path) -> None:
    (tmp_path / ".env").write_text(
        "FINANCIALASSIST_APP_NAME=local-service\nFINANCIALASSIST_DEBUG=true\n",
        encoding="utf-8",
    )

    settings = Settings()

    assert settings.app_name == "local-service"
    assert settings.debug is True


def test_environment_variables_override_dotenv(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    (tmp_path / ".env").write_text(
        "FINANCIALASSIST_APP_NAME=local-service\nFINANCIALASSIST_DEBUG=true\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("FINANCIALASSIST_APP_NAME", "environment-service")
    monkeypatch.setenv("FINANCIALASSIST_DEBUG", "false")

    settings = Settings()

    assert settings.app_name == "environment-service"
    assert settings.debug is False


def test_embedding_configuration_uses_expected_environment_variables(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("FINANCIALASSIST_EMBEDDING_MODEL", "test-embedding-model")

    settings = Settings()

    assert settings.openai_api_key == "test-key"
    assert settings.embedding_model == "test-embedding-model"


def test_agent_model_uses_environment_override(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FINANCIALASSIST_AGENT_MODEL", "test-agent-model")

    settings = Settings()

    assert settings.agent_model == "test-agent-model"
