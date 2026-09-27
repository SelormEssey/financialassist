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
