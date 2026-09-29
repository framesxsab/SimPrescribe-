# Contributing

SimpliScribe is a human-reviewed prescription aid. It does not diagnose, prescribe, or automatically substitute medicines. Never contribute patient identifiers, prescription images, credentials, or unsanitized logs.

## Set up

Install Git LFS before cloning, then run `git lfs install` and `git lfs pull`. Validate the two materialized CSV files before building indexes.

Use Python 3.11. Keep the WSL/Linux `.venv` separate from native Windows `.venv-windows`:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt -r requirements-dev.txt
cp .env.example .env
```

```powershell
py -3.11 -m venv .venv-windows
.\.venv-windows\Scripts\Activate.ps1
python -m pip install -r requirements.txt -r requirements-dev.txt
Copy-Item .env.example .env
```

Validate source data with `python scripts/validate_datasets.py`. For a fresh database, run `alembic upgrade head`, then build the required index with `python -m simpliscribe.build_lexicon_index`. `python -m simpliscribe.build_optional_reference_index` builds the optional references.

The bundled CSV datasets are CC BY-SA 4.0, separately from the MIT code. Read [dataset provenance](docs/DATASET_PROVENANCE.md) and [third-party notices](THIRD_PARTY_NOTICES.md) before changing or redistributing them. New data requires an exact source, version, matching checksum, and redistribution terms.

## Checks before a pull request

```bash
pytest -q
ruff check app.py simpliscribe tests
python -m simpliscribe.benchmark --cases data/golden_cases.v1.json --output data/benchmark_runs/local.json --min-f1 0.85 --max-hallucination-rate 0.10
```

For UI changes, run `npm ci` and `npm run test:a11y`. For schema changes, add a new Alembic migration and upgrade a fresh database. Use the [pull request checklist](.github/pull_request_template.md).
