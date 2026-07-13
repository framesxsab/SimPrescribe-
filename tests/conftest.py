import os
from pathlib import Path
import sys
import tempfile


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

TEST_DATABASE = Path(tempfile.gettempdir()) / f"simpliscribe-pytest-{os.getpid()}.db"
TEST_DATABASE.unlink(missing_ok=True)
os.environ["APP_ENV"] = "test"
os.environ["AUTH_REQUIRED"] = "false"
os.environ["DATABASE_URL"] = f"sqlite:///{TEST_DATABASE.as_posix()}"
os.environ["INFERENCE_PROVIDER"] = "fallback"


def pytest_sessionfinish(session, exitstatus):
    storage = sys.modules.get("simpliscribe.storage")
    if storage:
        storage.engine.dispose()
    TEST_DATABASE.unlink(missing_ok=True)
