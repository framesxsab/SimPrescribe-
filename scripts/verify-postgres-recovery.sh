#!/usr/bin/env bash
# Guarded PostgreSQL backup/restore drill. Restore URL must name a disposable
# database containing restore, verify, or test. Never points at production by name.
set -euo pipefail

SOURCE_URL=""
RESTORE_URL=""
BACKUP_PATH=""

usage() {
  echo "Usage: $0 --source-url URL --restore-url URL --backup-path PATH" >&2
  exit 1
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --source-url)
      SOURCE_URL="${2:-}"
      shift 2
      ;;
    --restore-url)
      RESTORE_URL="${2:-}"
      shift 2
      ;;
    --backup-path)
      BACKUP_PATH="${2:-}"
      shift 2
      ;;
    *)
      usage
      ;;
  esac
done

if [[ -z "$SOURCE_URL" || -z "$RESTORE_URL" || -z "$BACKUP_PATH" ]]; then
  usage
fi

for command in pg_dump pg_restore psql; do
  if ! command -v "$command" >/dev/null 2>&1; then
    echo "$command is required. Install PostgreSQL client tools before running recovery verification." >&2
    exit 1
  fi
done

if command -v python3 >/dev/null 2>&1; then
  PYTHON=python3
elif command -v python >/dev/null 2>&1; then
  PYTHON=python
else
  echo "python3 is required. Install PostgreSQL client tools before running recovery verification." >&2
  exit 1
fi

mapfile -t CONVERTED < <("$PYTHON" - "$SOURCE_URL" "$RESTORE_URL" "$BACKUP_PATH" <<'PY'
import sys
from pathlib import Path
from urllib.parse import unquote, urlparse

def convert(url: str) -> str:
    if url.startswith("postgresql+psycopg2:"):
        return "postgresql:" + url[len("postgresql+psycopg2:"):]
    if url.startswith("postgresql+psycopg:"):
        return "postgresql:" + url[len("postgresql+psycopg:"):]
    return url

def target(url: str) -> tuple[str, str]:
    parsed = urlparse(url)
    database_name = unquote((parsed.path or "").lstrip("/"))
    if not parsed.hostname or not database_name:
        raise SystemExit("Database URL must include a host and database name.")
    port = parsed.port or 5432
    identity = f"{parsed.hostname.lower()}:{port}/{database_name}"
    return identity, database_name

source_url, restore_url, backup_path = sys.argv[1], sys.argv[2], sys.argv[3]
source_identity, _source_name = target(convert(source_url))
restore_identity, restore_name = target(convert(restore_url))
if source_identity == restore_identity:
    raise SystemExit("Restore target must differ from source database.")
if not any(marker in restore_name.lower() for marker in ("restore", "verify", "test")):
    raise SystemExit("Restore target name must include restore, verify, or test. Refusing a potentially production target.")
path = Path(backup_path)
if path.exists():
    raise SystemExit("Backup path already exists. Choose a new immutable backup path instead of overwriting it.")
if not path.parent.exists():
    raise SystemExit("Backup directory does not exist.")
print(convert(source_url))
print(convert(restore_url))
PY
)

SOURCE_CONN="${CONVERTED[0]}"
RESTORE_CONN="${CONVERTED[1]}"

pg_dump --format=custom --file="$BACKUP_PATH" "$SOURCE_CONN"
pg_restore --list "$BACKUP_PATH" >/dev/null
# ponytail: restore only accepts an explicitly named disposable target; use a managed restore workflow if finer-grained recovery is needed.
pg_restore --clean --if-exists --no-owner --no-privileges --dbname="$RESTORE_CONN" "$BACKUP_PATH"

result="$(psql "$RESTORE_CONN" --tuples-only --no-align --command "SELECT CASE WHEN to_regclass('public.analyses') IS NULL THEN 'missing' ELSE 'ready' END;")"
if [[ "$result" != *ready* ]]; then
  echo "Restored database did not contain the analyses table." >&2
  exit 1
fi

echo "Recovery verification passed. Preserve the backup in encrypted managed storage and record the verification date."
