from __future__ import annotations

import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from simpliscribe.dataset_validator import validate_all_datasets


def main() -> None:
    print("=" * 60)
    print("SIMPLISCRIBE PRODUCTION DATASET INTEGRITY CHECK")
    print("=" * 60)

    summary = validate_all_datasets()
    for ds in summary["datasets"]:
        status_symbol = "[OK]" if ds["valid"] else "[FAIL]"
        print(f"\n{status_symbol} {ds['name']}")
        print(f"   Path:     {ds['path']}")
        print(f"   Exists:   {ds['exists']} ({ds['size_mb']} MB)")
        print(f"   Rows:     {ds['row_count']:,}")
        print(f"   Checksum: {ds['sha256_prefix']}...")
        if ds["errors"]:
            for err in ds["errors"]:
                print(f"   ERROR:    {err}")

    print("\n" + "=" * 60)
    if summary["all_valid"]:
        print("ALL DATASETS PASSED INTEGRITY & SCHEMA CHECKS.")
        print("=" * 60 + "\n")
        sys.exit(0)
    else:
        print("DATASET VALIDATION FAILED! SEE ERRORS ABOVE.")
        print("=" * 60 + "\n")
        sys.exit(1)


if __name__ == "__main__":
    main()
