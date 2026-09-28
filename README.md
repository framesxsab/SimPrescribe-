---
title: SimpliScribe
sdk: docker
app_port: 7860
pinned: false
---

# SimpliScribe

SimpliScribe is a FastAPI application that simplifies prescription reading by extracting text from prescription images or PDFs and turning that OCR output into a structured medication summary.

The patient upload now persists the private source before OCR, then opens a dedicated processing page that reads saved backend stages while OCR and structuring run. The analysis records `uploaded`, `processing`, `review_required`, `confirmed`, or `processing_failed`; OCR output is checkpointed before structuring. Patients can retry a failed stage from the saved prescription, save review edits before confirmation, and refresh the processing page without starting another analysis. Pharmacy matching requires confirmation. Required medicine aliases load from the fingerprinted SQLite index when it is current; build or refresh it with `python -m simpliscribe.build_lexicon_index`. A missing or stale index falls back to authoritative CSV construction, and an existing stale index is logged. The generated SQLite database is a rebuildable runtime artifact and is not committed. Optional substitutes, uses, and side effects are read by targeted name lookup from a separate fingerprinted SQLite index when details or reports are requested; composition remains part of the required lexicon data. Build the optional index with `python -m simpliscribe.build_optional_reference_index`. The generated database is a rebuildable, ignored runtime artifact. If it is missing, stale, or unavailable, core prescription details still render and the page labels optional reference information unavailable. Optional reference data is not loaded during upload, OCR, review, or confirmation. PDF reports are generated on demand; optional web/model alternative lookup does not delay review. The existing `/api/analyze` endpoint remains synchronous for API compatibility.

The OCR reader warms once in the background after application startup in non-test processes. `/api/live` stays available during warmup; `/api/health` reports OCR readiness. Uploads that arrive during warmup remain attached to their saved processing record and wait for the same reader instance.

It also provides an authenticated prescription-first marketplace MVP: patients can confirm the structured result, find approved local pharmacies serving their PIN code, request a pharmacist-verified quote, and track COD pickup or local-delivery fulfillment. Pharmacies manage exact-name inventory and order requests through a dedicated portal.

## Safety and intended use

SimpliScribe is an open-source **review aid**, not a prescribing, diagnosis, dispensing, or autonomous clinical decision system. Every medicine name, strength, route, frequency, duration, interaction, and dataset reference candidate must be checked against the original prescription by a qualified clinician or pharmacist. The committed golden gate has ten synthetic cases and is a regression check, not evidence of clinical accuracy.

**The platform helps patients understand and fulfil an existing prescription; it does not perform medical diagnosis.** Generic/equivalent entries are informational reference candidates and cannot replace an ordered prescription line automatically.

The application preserves OCR line boundaries, exposes OCR confidence and provider provenance, marks uncertain fields for review, validates actual file content, deletes processing copies after OCR while retaining a protected source for pharmacist verification, and labels dataset alternatives as reference candidates rather than recommendations.

Analyses are stored through SQLAlchemy. Local development defaults to SQLite and can be unauthenticated, so it is strictly a one-user, local-only workflow. Do not expose it to remote or multi-user healthcare traffic. Identifiable data needs authenticated role-based access, encrypted managed storage, audit/retention controls, a threat model, licensed/versioned medicine sources, and prospective clinical validation.

## Open-source local stack

- OCR runs locally with PaddleOCR.
- Medication structuring can run with `INFERENCE_PROVIDER=fallback` or a self-hosted endpoint via `INFERENCE_PROVIDER=endpoint`.
- No external paid API is required if you keep OCR local and point the endpoint mode at your own deployed model.

## Local model server

This repo includes a separate local model server that you can run on your own laptop and point the main app to.

Install the optional model-serving dependencies:

```bash
pip install -r requirements-local-model.txt
```

Start the model server on port `8001`:

```bash
uvicorn simpliscribe.local_model_server:app --host 127.0.0.1 --port 8001
```

Then point the main app at that local endpoint:

```env
INFERENCE_PROVIDER=endpoint
MODEL_API_URL=http://127.0.0.1:8001/extract
LOCAL_MODEL_ID=Qwen/Qwen2.5-1.5B-Instruct
LOCAL_MODEL_DEVICE=auto
LOCAL_MODEL_TEMPERATURE=0.1
LOCAL_MODEL_MAX_NEW_TOKENS=256
```

Set `MODEL_SERVER_API_KEY=<shared-secret>` on the model server and the matching
`MODEL_API_KEY=<shared-secret>` on the main app to authenticate `/extract` calls
(optional but recommended before exposing the model server beyond 127.0.0.1).
`MODEL_SERVER_MAX_INPUT_CHARS` caps the accepted input and prompt length.

The default local model is intentionally small enough to be more realistic on consumer hardware. If you have a stronger GPU, you can raise `LOCAL_MODEL_ID` to a larger open model.

For a 6 GB GPU, `Qwen/Qwen2.5-1.5B-Instruct` is the recommended default starting point before trying larger models.
The first local request can take a few minutes because model weights may need to download and load into memory. If that tradeoff is acceptable on a trusted local machine, set `REQUEST_TIMEOUT_SECONDS=300` temporarily; remote deployments should keep a short bounded timeout.

## Runtime options

- `INFERENCE_PROVIDER=fallback`
  Uses a local heuristic fallback and does not require external model credentials.
- `INFERENCE_PROVIDER=huggingface`
  Uses the Hugging Face Inference API with `HUGGINGFACEHUB_API_TOKEN` and `HF_CHAT_MODEL`.
- `INFERENCE_PROVIDER=endpoint`
  Sends OCR text to a compatible HTTP endpoint using `MODEL_API_URL` and optional `MODEL_API_KEY`.

### Alternative medicine reference candidates

When a listed medicine may be unavailable, SimpliScribe first looks in the **bundled trained datasets**: CSV substitute columns and other brands that share the same composition. That local lookup does not send data off the machine.

If the local list is empty, the optional web/model lookup helper can add more **reference candidates** from a configured model or DuckDuckGo. The patient upload path does not invoke this helper during core structuring, even when enabled. It is **off by default** (fail-closed) because it sends data outside the box:

```env
ALTERNATIVES_ENABLED=true
ALTERNATIVES_PROVIDER=auto      # auto = model first, then DuckDuckGo; also: model, web, duckduckgo
ALTERNATIVES_TIMEOUT_SECONDS=15
ALTERNATIVES_CACHE_TTL_SECONDS=86400
ALTERNATIVES_MAX_CANDIDATES=5
```

Governance and safety:

- Only the canonical medicine name is ever sent to the model or web tier. Patient
  names, doctor names, and raw OCR text never leave the server through this path.
- Candidates are validated against the bundled local datasets before display, so a
  hallucinated or non-existent drug name cannot be surfaced. The India dataset is
  brand-centric, so generic names that do not appear in it (for example bare
  "Ibuprofen" or "Amoxicillin") are filtered out even when a model returns them.
- On-screen and PDF reports label local hits as "If this medicine is unavailable
  (local dataset)" and off-box hits as "If this medicine is unavailable (web/model)".
  They are never recommendations; they force `requires_review` and carry source
  links where available.
- Lookups are TTL-cached, capped per analysis, bounded by a timeout, and any error
  fails open to an empty list so the extraction pipeline never breaks.

## Local development

### Clean checkout prerequisites

Use Python 3.11 for the full pinned runtime (the container uses 3.11.9; Windows RC validation used 3.11.15). The lightweight package metadata permits Python 3.10 or newer, but is not the complete native OCR environment. Local Python must include `venv`/`ensurepip` support; on Ubuntu install the matching `python3.X-venv` OS package when `python -m venv` reports that `ensurepip` is missing. Runtime packages are pinned in `requirements.txt`; development lint/coverage packages are in `requirements-dev.txt`. Native Linux also needs `libgl1`, `libglib2.0-0`, `libsm6`, `libxext6`, `libxrender1`, and `libgomp1` (the Dockerfile installs these). Node.js/npm are only needed for the Playwright suites. On Ubuntu, install Chromium system libraries with `npx playwright install-deps chromium` before running those suites. The full pytest suite also invokes the PostgreSQL recovery guard tests, so `pg_dump`, `pg_restore`, and `psql` from the PostgreSQL client package must be on `PATH`; the actual restore drill additionally needs a disposable PostgreSQL server.

- **Source-controlled:** application code, migrations, `.env.example`, and the two medicine CSV datasets. The CSV files (`A_Z_medicines_dataset_of_India.csv`, `all_medicine databased.csv`) are Git LFS objects. Install Git LFS before cloning, run `git lfs install`, then clone and run `git lfs pull`. Without LFS, Git supplies pointer text instead of CSV data; validate with `python scripts/validate_datasets.py` before continuing.
- **Generated at build/setup:** `data/medicine_lexicon.sqlite` is required and `data/medicine_optional_references.sqlite` is optional. Both are fingerprinted SQLite indexes generated from the LFS CSVs; neither is committed. Local setup commands below build them; Docker builds them into the image.
- **Generated at runtime:** the local application database is created by `alembic upgrade head` (`0002_marketplace`); protected uploads are written under `data/prescriptions`; temporary upload copies go under `uploads`; OCR models are downloaded on first initialization to `tmp/ocr-cache`. Paddle and PaddleX default to subdirectories beneath that cache via `PADDLE_HOME` and `PADDLE_PDX_CACHE_HOME`. The first uncached OCR initialization needs network access to the model hosting endpoints. `INFERENCE_PROVIDER=fallback` avoids inference API credentials, not OCR models.
- **Secret configuration:** `.env.example` contains development-only synthetic values. Copy it to an untracked `.env` for local setup; production must use unique secrets from its secret manager and must not use `.env.example`.

### Linux / WSL / macOS

Run from the repository root. Keep this environment separate from native Windows:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt -r requirements-dev.txt
cp .env.example .env
```

On a minimal Linux host where `ensurepip` is unavailable and installing the matching OS `venv` package is not possible, an existing modern system `pip` can bootstrap the isolated environment: `python3.11 -m venv --without-pip .venv`, then `python3.11 -m pip --python .venv/bin/python install -r requirements.txt -r requirements-dev.txt`. Activate the environment and copy `.env.example` as above.

### Windows PowerShell

Use native 64-bit Python 3.11 and a separate Windows environment. Do not activate or reuse a `.venv` created by WSL/Linux: its `pyvenv.cfg` may point to `/usr/bin`.

```powershell
py -3.11 -m venv .venv-windows
.\.venv-windows\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt -r requirements-dev.txt
Copy-Item .env.example .env
```

For an existing uv-managed Python installation, `uv venv --python 3.11 --seed .venv-windows` replaces the first command. If activation is unavailable, use `.\.venv-windows\Scripts\python.exe -m ...` explicitly. The Windows RC was validated with Python 3.11.15, PaddlePaddle 3.2.2, PaddleOCR 3.4.0, PaddleX 3.4.3, NumPy 2.4.6, and only `opencv-contrib-python` 4.10.0.84. These native runtime packages are pinned in `requirements.txt`, which Docker and CI also install; `pyproject.toml` alone does not install the OCR runtime. Optional local-model dependencies remain in `requirements-local-model.txt`.

Set `INFERENCE_PROVIDER=fallback` in the local `.env` for a credential-free local run. Keep `.env`, virtual environments, databases, uploads, and OCR caches out of source control. `.env.example` supplies development-only local values and must not be used as a production secret file.

Run the clean-database migration and generate both local indexes before starting:

```bash
alembic upgrade head
alembic current
python scripts/validate_datasets.py
python -m simpliscribe.build_lexicon_index
python -m simpliscribe.build_optional_reference_index  # optional
uvicorn app:app --reload
```

The database starts empty and must be migrated to Alembic head (`0002_marketplace`) before use. Development and test startup retain `ensure_schema()` as a compatibility bootstrap. Production startup does not create schema; run `alembic upgrade head` before starting the application. `DATABASE_URL`, `SESSION_SECRET`, `INFERENCE_PROVIDER`, `OCR_CACHE_DIR`, and retention values can be set in `.env`; see `.env.example` for the complete supported configuration. A shared/production deployment must set a unique `SESSION_SECRET` of at least 32 characters, PostgreSQL `DATABASE_URL`, and authentication configuration as described in [Production safety configuration](#production-safety-configuration).

Create an approved demo pharmacy and starter exact-name inventory after initializing the database:

```powershell
python scripts\seed_marketplace.py --password "choose-a-demo-password" --pin 400001
```

Patient accounts register at `/register/patient`; pharmacies register at `/register/pharmacy` and require approval at `/admin/pharmacies`.
Patients can open `/marketplace` from the top navigation, choose an analyzed prescription, confirm its medicines, and find pharmacies serving their PIN. The marketplace needs a signed-in patient and at least one approved pharmacy to show results.

```bash
uvicorn app:app --reload
```

Recommended local environment variables for a fully self-hosted setup:

```env
INFERENCE_PROVIDER=endpoint
MODEL_API_URL=http://127.0.0.1:8001/extract
OCR_LANGUAGE=en
OCR_USE_GPU=false
OCR_CACHE_DIR=./tmp/ocr-cache
# PaddleX 3.4 honors PADDLE_PDX_CACHE_HOME; Paddle 3.3 honors PADDLE_HOME.
# Leave these unset to derive both locations beneath OCR_CACHE_DIR.
# PADDLE_HOME=./tmp/ocr-cache/paddle
# PADDLE_PDX_CACHE_HOME=./tmp/ocr-cache/paddlex
```

Relative OCR cache paths resolve from the repository root, independent of the shell or server working directory. Local CPU OCR still takes tens of seconds per image on measured runs; background warmup removes model initialization from a first upload when warmup completes in advance, but does not reduce inference time. OCR cache directories are created and write-tested at startup. Paddle/PaddleX model binaries are runtime data and must stay outside the repository (the default `tmp/ocr-cache` is ignored). PaddleOCR imports and initializes with its supported cache variables and legacy home expansion routed to the configured directory; the original process environment is restored afterward. Absolute `OCR_CACHE_DIR`, `PADDLE_HOME`, and `PADDLE_PDX_CACHE_HOME` overrides are preserved.

If you do not want to run the local model server yet, keep `INFERENCE_PROVIDER=fallback` and the app will stay fully local with rule-based extraction only.

Open `http://127.0.0.1:8000`.

## Testing

```bash
pytest
```

The axe accessibility suite uses Playwright. With the app running and an existing local patient account, install its dev tools and browser once, set the account and any saved records to scan, then run it:

```powershell
npm ci
npx playwright install chromium
$env:SIMPLISCRIBE_BASE_URL = "http://127.0.0.1:8000"
$env:A11Y_EMAIL = "local-patient@example.test"
$env:A11Y_PASSWORD = "the-local-test-password"
# Optional IDs for records owned by that patient:
$env:A11Y_PROCESSING_ID = "..."
$env:A11Y_REVIEW_ID = "..."
$env:A11Y_DETAILS_ID = "..."
$env:A11Y_ORDER_ID = "..."
npm run test:a11y
```

CI (`.github/workflows/quality.yml`) also checks that `uvicorn app:app` can import the ASGI app, applies Alembic migrations to a throwaway SQLite database, runs a ruff check baseline (no autoformat), publishes a pytest coverage report artifact, builds the Docker image without preloading OCR models, restores a disposable PostgreSQL database in CI, and runs the synthetic golden gate below.

## Pipeline fallbacks and health

Internal failures should still return a complete analysis shape rather than a partial unlabeled payload:

- Hugging Face / HTTP endpoint errors fall back to the rule-based parser, with `pipeline.used_provider`, `pipeline.warnings`, and `human_review_required`.
- If the heuristic parser or lexicon also fails, the API still returns `N/A` headers and `medications: []` plus `pipeline.degraded` and an `error_code`.
- Unreadable scans stay `422 UNUSABLE_PRESCRIPTION`. Storage or PDF failures return `503` with `STORAGE_FAILED` or `REPORT_UNAVAILABLE`.
- `/api/live` is a cheap process check for the container HEALTHCHECK. `/api/health` reports database, required lexicon-index, OCR, dataset, and inference-provider readiness; it reports `degraded` while any required component is unavailable. Logs use `error_code` / `analysis_id` / provider and must not include raw OCR or patient names.

## Benchmarking

You can benchmark extraction quality locally against curated OCR text cases or full image/PDF cases.

Run the benchmark with the current provider configuration:

```bash
python -m simpliscribe.benchmark --cases data/benchmark_cases.sample.json
```

This writes a JSON report to `data/benchmark_runs/latest.json` and prints a summary score in the terminal.

Case files support either:

- `raw_text`: benchmark only the structuring stage
- `file_path`: benchmark the full OCR + structuring pipeline
- `.parquet` with a `ground_truth` column: auto-converted into synthetic prescription benchmark cases

Parquet input requires `pandas` and `pyarrow` in the active environment.

Example parquet benchmark run:

```bash
python -m simpliscribe.benchmark --cases 0000.parquet --limit 25 --output data/benchmark_runs/parquet_0000.json
```

Use `--limit` while iterating on larger parquet datasets so the benchmark stays fast enough to compare fallback and endpoint modes.

Example file-based case shape:

```json
[
  {
    "id": "scan-1",
    "label": "Prescription image",
    "file_path": "../uploads/prescription.png",
    "expected_medications": [
      {
        "name": "Paracetamol",
        "type": "Tablet",
        "dosage": "650 mg",
        "frequency": "once daily",
        "duration": "5 days"
      }
    ]
  }
]
```

Recommended workflow:

1. Add real OCR text samples to `data/benchmark_cases.sample.json` or a separate JSON file.
2. Add file-based cases when you want to measure OCR and structuring together.
3. Run once with `INFERENCE_PROVIDER=fallback`.
4. Run again with `INFERENCE_PROVIDER=endpoint` and your local model server.
5. Compare the saved benchmark reports before changing prompts or models.

### Versioned golden regression set

The committed `data/golden_cases.v1.json` file uses schema version `1.0` and covers clean, multi-medication, timing, missing-field, look-alike, false-positive, and unreadable synthetic OCR scenarios. It is deliberately labelled synthetic and must not be presented as clinical validation. Do not replace this file.

Clinician-adjudicated cases, when they exist, belong only in `data/golden_cases.clinician.v1.json` (currently an empty schema `1.0` placeholder). Append cases there; never copy over `golden_cases.v1.json`. Optional local merge:

```bash
INFERENCE_PROVIDER=fallback python -m simpliscribe.benchmark \
  --cases data/golden_cases.v1.json \
  --extra-cases data/golden_cases.clinician.v1.json \
  --output data/benchmark_runs/latest.json \
  --min-f1 0.85 \
  --max-hallucination-rate 0.10
```

Run the same quality gate used by CI:

```bash
INFERENCE_PROVIDER=fallback python -m simpliscribe.benchmark \
  --cases data/golden_cases.v1.json \
  --output data/benchmark_runs/latest.json \
  --min-f1 0.85 \
  --max-hallucination-rate 0.10
```

On PowerShell, set `$env:INFERENCE_PROVIDER="fallback"` before running the Python command. Reports include medicine-name precision/recall/F1, hallucination rate, accuracy by extracted field, expected-review flag recall, and unreadable-input rejection rate. A failed threshold exits non-zero for CI.

Golden files accept a top-level object containing `schema_version`, provenance metadata, and `cases`. Each case requires a unique `id`, an `expected_medications` array, and either `raw_text` or a relative `file_path`. Optional `tags`, per-medication `requires_review`, and case-level `expected_rejection` fields enable subgroup and safety evaluation. Reviewed image fixtures should be de-identified and committed only when consent and dataset terms allow it.

For a clinically meaningful evaluation, add de-identified, consented prescriptions adjudicated by qualified reviewers to `data/golden_cases.clinician.v1.json` without replacing the synthetic set. Report medication-name precision/recall, exact strength/frequency/duration accuracy, unreadable-scan rejection, subgroup performance, and false confident matches. Do not promote a model based on one aggregate score.

## Practical roadmap

1. Build a versioned golden set with reviewer agreement and look-alike/sound-alike cases.
2. Compare every OCR or model proposal on that same set and adopt only measured improvements.
3. Replace unverified CSV provenance with licensed, versioned sources and stable medicine identifiers. Until then, keep the committed CSVs and record what is known in [docs/DATASET_PROVENANCE.md](docs/DATASET_PROVENANCE.md).
4. Prefer OpenID Connect for shared reviewer access; keep the bootstrap admin account as emergency access only. See [docs/CONSENT_AND_RETENTION.md](docs/CONSENT_AND_RETENTION.md).
5. Complete operational security, accessibility, workflow, and prospective clinical validation before production use. Start from [docs/simpliscribe-threat-model.md](docs/simpliscribe-threat-model.md).

### Shipped workflow safeguards

Requirements reviewed from supplied course material were distilled without copying personal details. The following can improve this review aid without turning it into a diagnosis or prescribing system:

1. **Role-scoped reviewer access:** bootstrap deployments can assign `admin`, `reviewer`, or read-only `auditor`; edit routes enforce reviewer/admin server-side. Optional OpenID Connect sign-in maps immutable provider subjects to admin/reviewer roles and defaults all others to read-only auditor access.
2. **Final-report integrity:** each review preserves the prior medication and review state as a numbered version, rejects stale concurrent updates, emits an audit event, and exposes owner-scoped audit retrieval at `/api/audit`; use managed database migrations before independently evolving deployed versions.
3. **Operational recovery:** the guarded [PostgreSQL recovery verifier](docs/PRODUCTION_RECOVERY.md) can verify a backup against a disposable private restore database; record the result in the approved operations system before retaining identifiable data.
4. **Accessible report output:** PDF and on-screen reports are readable, printable, explicit about human verification, and show how many prior review states are preserved.
5. **Unavailable-medicine reference list:** local CSV substitutes and same-composition brands are shown first. The optional web/model helper requires an explicit call outside the core upload path and `ALTERNATIVES_ENABLED=true`.
6. **Degraded analysis output:** model, OCR-engine, lexicon, database, and PDF failures keep a labeled payload or a documented error code instead of a silent partial result.

These are technical safeguards, not clinical validation. Patient and approved-pharmacy accounts support prescription fulfillment only; SimpliScribe does not add consultation/diagnosis records, automatic treatment decisions, medicine reminders, or drug-interaction decisioning.

Code is MIT licensed. Dataset files may have separate upstream terms; see [docs/DATASET_PROVENANCE.md](docs/DATASET_PROVENANCE.md) before redistribution.

## Docker deployment

### Production safety configuration

Production mode fails closed unless authentication, a strong session secret, and a non-SQLite database are configured. Use a managed PostgreSQL service with encryption at rest, backups, private networking, and TLS enforcement. Store all values below in the deployment platform's secret manager rather than committing an `.env` file.

```bash
APP_ENV=production
DATABASE_URL=postgresql+psycopg://USER:PASSWORD@HOST:5432/simpliscribe?sslmode=require
SESSION_SECRET=<at-least-32-random-characters>
ADMIN_EMAIL=reviewer@example.com
ADMIN_PASSWORD=<strong-secret-manager-value>
RETENTION_DAYS=30
SESSION_MAX_AGE_SECONDS=28800
SESSION_HTTPS_ONLY=true
INFERENCE_PROVIDER=fallback
REQUEST_TIMEOUT_SECONDS=60
# Rate limiting keys on the first X-Forwarded-For hop instead of the proxy IP.
# Enable ONLY behind a trusted proxy that strips inbound X-Forwarded-For
# (otherwise the header is spoofable and the rate limit can be bypassed).
TRUST_PROXY_HEADERS=false
```

For managed identity, replace bootstrap credentials with these deployment secrets:

```bash
OIDC_ISSUER=https://identity.example.com
OIDC_CLIENT_ID=<client-id>
OIDC_CLIENT_SECRET=<secret-manager-value>
OIDC_REDIRECT_URI=https://app.example.com/auth/callback
OIDC_ADMIN_SUBJECTS=<comma-separated-provider-subject-ids>
OIDC_REVIEWER_SUBJECTS=<comma-separated-provider-subject-ids>
```

Production behavior includes signed HTTP-only cookies, CSRF validation, explicit upload consent, automatic analysis expiry, redacted audit events, protected history/details/reports/review APIs, analysis concurrency limits, CSP/HSTS headers, and a non-root container liveness check. OIDC users map to admin/reviewer/auditor roles; unmapped users are read-only auditors.

The configured administrator is a local bootstrap account. Prefer OIDC before onboarding multiple reviewers; keep bootstrap credentials in the secret manager for emergency access. Apply schema changes through the managed Alembic migrations below, not ad-hoc DDL, before independently evolving multiple deployed versions.

### Database migrations (Alembic)

Schema changes are managed with Alembic. Run the migration step before starting a new deployment; it is idempotent against an empty or up-to-date database.

```bash
# Upgrade to the latest schema (uses the same DATABASE_URL as the app)
alembic upgrade head

# Inspect the current revision
alembic current
```

Development and test startup retain `ensure_schema()` as a compatibility bootstrap. Production startup does not create schema; run `alembic upgrade head` as part of each release and evolve the schema by adding a new revision (`alembic revision --autogenerate -m "describe change"`) rather than editing existing ones.


The review screen supports correction, confirmation, unreadable rejection, and sign-out for shared workstations. Processing copies are removed after OCR; a protected source copy remains only for the patient and authorized pharmacist workflow until retention expiry. Do not enable identifiable patient uploads until the deployment has a documented consent basis, retention owner, incident process, backup/restore test, threat model, and approved medicine-dataset licensing. See [docs/CONSENT_AND_RETENTION.md](docs/CONSENT_AND_RETENTION.md) and [docs/simpliscribe-threat-model.md](docs/simpliscribe-threat-model.md). Configure request-size limits at the ingress/proxy as well as `MAX_UPLOAD_MB`; multipart bodies reach the server before application validation.

```bash
docker build -t simpliscribe .
docker run --rm --env-file production.env simpliscribe alembic upgrade head
docker run --rm -p 127.0.0.1:7860:7860 --env-file production.env simpliscribe
```

The Docker image sets `OCR_CACHE_DIR`, `PADDLE_HOME`, and `PADDLE_PDX_CACHE_HOME` to the same writable `/app/tmp/ocr-cache` tree during model preload and runtime. The Docker build generates both fingerprinted medicine indexes into the image after copying the LFS datasets. A fresh named `/app/data` volume is initialized from that image content. CI builds the image with `--build-arg PRELOAD_OCR=0` so the job does not download OCR weights. Production images should keep the default `PRELOAD_OCR=1`.

The image defaults to `APP_ENV=production` and fails closed without the complete production configuration above. Keep `production.env` outside the repository and secret manager values out of `.env.example`. Production session cookies require HTTPS: keep the container bound to loopback and place a TLS-terminating reverse proxy in front of `http://127.0.0.1:7860`; do not expose or browse the raw HTTP port directly. Use the local `uvicorn` workflow, not a remotely reachable Docker container, for unauthenticated development.

For the bundled Compose stack, set `POSTGRES_PASSWORD` to a URL-safe secret, `SESSION_SECRET` to a unique value of at least 32 characters, and `ADMIN_EMAIL`/`ADMIN_PASSWORD` from a secret manager. The file has no credential defaults. The bundled Nginx listener binds only to `127.0.0.1:8080`; terminate HTTPS in a separate trusted ingress before routing traffic there. Migrate the fresh database before starting the app:

```bash
docker compose -f docker-compose.prod.yml up -d db
docker compose -f docker-compose.prod.yml run --rm web alembic upgrade head
docker compose -f docker-compose.prod.yml up -d web nginx
```

## Hugging Face Spaces

Do not deploy this live upload workflow to a public Hugging Face Space. A public Space cannot provide the required access controls, operational guarantees, or data-governance review. A public showcase must be static/synthetic with uploads disabled; use a private, authenticated deployment with managed PostgreSQL for any real workflow.

For a controlled private demo:

1. Create a private Docker Space.
2. Choose `Docker` as the SDK.
3. Configure every value from the production safety block as a Space secret, including managed PostgreSQL, session secret, and bootstrap credential.
4. Keep `INFERENCE_PROVIDER=fallback` for a local-only inference path, or use a processor approved for the data you submit.

Example git remote setup:

```bash
git remote add space https://huggingface.co/spaces/fxsab/simpliscribe
git push space master
```

Expected behavior on Spaces:

- Keep the Space private and use synthetic/de-identified input only.
- OCR runs inside the Space container.
- No paid model API is required when `INFERENCE_PROVIDER=fallback`.
- CPU performance and persistent-storage guarantees depend on the selected Space hardware.

For an external hosted model, set `INFERENCE_PROVIDER=endpoint` and point `MODEL_API_URL` at an endpoint approved to receive prescription OCR text.
