from pathlib import Path

import multimodal_rag.config as config


def test_should_trust_system_certs_early_reads_profile_env_file(
    monkeypatch,
    tmp_path: Path,
) -> None:
    (tmp_path / ".env.local").write_text("TRUST_SYSTEM_CERTS=true\n", encoding="utf-8")
    monkeypatch.delenv("TRUST_SYSTEM_CERTS", raising=False)
    monkeypatch.setenv("RAG_ENV", "local")
    monkeypatch.setattr(config, "PROJECT_ROOT", tmp_path)

    assert config._should_trust_system_certs_early() is True


def test_should_trust_system_certs_early_prefers_os_env_var(
    monkeypatch,
    tmp_path: Path,
) -> None:
    (tmp_path / ".env.local").write_text("TRUST_SYSTEM_CERTS=false\n", encoding="utf-8")
    monkeypatch.setenv("TRUST_SYSTEM_CERTS", "1")
    monkeypatch.setenv("RAG_ENV", "local")
    monkeypatch.setattr(config, "PROJECT_ROOT", tmp_path)

    assert config._should_trust_system_certs_early() is True
