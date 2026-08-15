import os
import shutil
import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
PS1 = REPO / "scripts" / "verify-postgres-recovery.ps1"
SH = REPO / "scripts" / "verify-postgres-recovery.sh"


def _verifier_commands(source: str, restore: str, backup: str) -> list[list[str]]:
    commands: list[list[str]] = []
    if os.name == "nt" and shutil.which("powershell"):
        commands.append(
            [
                "powershell",
                "-NoProfile",
                "-File",
                str(PS1),
                "-SourceDatabaseUrl",
                source,
                "-RestoreDatabaseUrl",
                restore,
                "-BackupPath",
                backup,
            ]
        )
    if os.name != "nt" and shutil.which("bash") and SH.exists():
        commands.append(
            [
                "bash",
                str(SH),
                "--source-url",
                source,
                "--restore-url",
                restore,
                "--backup-path",
                backup,
            ]
        )
    if not commands:
        pytest.skip("No recovery verifier interpreter is available.")
    return commands


def test_recovery_verifier_refuses_the_source_database_as_restore_target(tmp_path):
    source = "postgresql://user:password@example.test/source"
    backup = str(tmp_path / "should-not-be-created.dump")
    for command in _verifier_commands(source, source, backup):
        result = subprocess.run(command, capture_output=True, text=True, check=False)
        assert result.returncode != 0
        assert "Restore target must differ from source database." in result.stderr
        assert not Path(backup).exists()


def test_recovery_verifier_requires_marker_in_database_name(tmp_path):
    backup = str(tmp_path / "should-not-be-created.dump")
    for command in _verifier_commands(
        "postgresql://user@example.test/source",
        "postgresql://test-user@example.test/production",
        backup,
    ):
        result = subprocess.run(command, capture_output=True, text=True, check=False)
        assert result.returncode != 0
        assert "Restore target name must include restore, verify, or test." in result.stderr
        assert not Path(backup).exists()
