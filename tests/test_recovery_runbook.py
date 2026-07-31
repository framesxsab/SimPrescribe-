import subprocess
from pathlib import Path


def test_recovery_verifier_refuses_the_source_database_as_restore_target():
    script = Path(__file__).resolve().parents[1] / "scripts" / "verify-postgres-recovery.ps1"
    source = "postgresql://user:password@example.test/source"
    result = subprocess.run(
        [
            "powershell",
            "-NoProfile",
            "-File",
            str(script),
            "-SourceDatabaseUrl",
            source,
            "-RestoreDatabaseUrl",
            source,
            "-BackupPath",
            str(Path.cwd() / "should-not-be-created.dump"),
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode != 0
    assert "Restore target must differ from source database." in result.stderr


def test_recovery_verifier_requires_marker_in_database_name():
    script = Path(__file__).resolve().parents[1] / "scripts" / "verify-postgres-recovery.ps1"
    result = subprocess.run(
        [
            "powershell",
            "-NoProfile",
            "-File",
            str(script),
            "-SourceDatabaseUrl",
            "postgresql://user@example.test/source",
            "-RestoreDatabaseUrl",
            "postgresql://test-user@example.test/production",
            "-BackupPath",
            str(Path.cwd() / "should-not-be-created.dump"),
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode != 0
    assert "Restore target name must include restore, verify, or test." in result.stderr
