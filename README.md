---
title: SimpliScribe
sdk: docker
app_port: 7860
pinned: false
---

# SimpliScribe

SimpliScribe is a FastAPI application that simplifies prescription reading by extracting text from prescription images or PDFs and turning that OCR output into a structured medication summary.

## Safety and intended use

SimpliScribe is an open-source **review aid**, not a prescribing, diagnosis, dispensing, or autonomous clinical decision system. Every medicine name, strength, route, frequency, duration, interaction, and dataset reference candidate must be checked against the original prescription by a qualified clinician or pharmacist. The committed golden gate has eight synthetic cases and is a regression check, not evidence of clinical accuracy.

The application preserves OCR line boundaries, exposes OCR confidence and provider provenance, marks uncertain fields for review, validates actual file content, deletes uploads after processing, and labels dataset alternatives as reference candidates rather than recommendations.

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

## Local development

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
copy .env.example .env
uvicorn app:app --reload
```

Recommended local environment variables for a fully self-hosted setup:

```env
INFERENCE_PROVIDER=endpoint
MODEL_API_URL=http://127.0.0.1:8001/extract
OCR_LANGUAGE=en
OCR_USE_GPU=false
```

If you do not want to run the local model server yet, keep `INFERENCE_PROVIDER=fallback` and the app will stay fully local with rule-based extraction only.

Open `http://127.0.0.1:8000`.

## Testing

```bash
pytest
```

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

The committed `data/golden_cases.v1.json` file uses schema version `1.0` and covers clean, multi-medication, timing, missing-field, false-positive, and unreadable synthetic OCR scenarios. It is deliberately labelled synthetic and must not be presented as clinical validation.

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

For a clinically meaningful evaluation, replace the samples with de-identified, consented prescriptions adjudicated by qualified reviewers. Report medication-name precision/recall, exact strength/frequency/duration accuracy, unreadable-scan rejection, subgroup performance, and false confident matches. Do not promote a model based on one aggregate score.

## Practical roadmap

1. Build a versioned golden set with reviewer agreement and look-alike/sound-alike cases.
2. Compare every OCR or model proposal on that same set and adopt only measured improvements.
3. Replace unverified CSV provenance with licensed, versioned sources and stable medicine identifiers.
4. Replace the bootstrap-admin login with external identity, role/tenant claims, and managed migrations before onboarding multiple reviewers.
5. Complete operational security, accessibility, workflow, and prospective clinical validation before production use.

Code is MIT licensed. Dataset files may have separate upstream terms; verify and document those terms before redistribution.

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
```

Production behavior includes a single bootstrap-admin session, signed HTTP-only cookies, CSRF validation, explicit upload consent, automatic analysis expiry, redacted audit events, protected history/details/reports/review APIs, analysis concurrency limits, CSP/HSTS headers, and a non-root container liveness check. It is not yet a multi-user identity or RBAC system.

The configured administrator is a deployment bootstrap account. For an organization or multiple reviewers, replace it with an external identity provider before onboarding users. Database creation currently uses SQLAlchemy metadata initialization; adopt managed migrations before independently evolving multiple deployed versions.

The review screen supports correction, confirmation, unreadable rejection, and sign-out for shared workstations. Original uploads are removed after processing, so reviewers must compare against their own source document during the active workflow. Do not enable identifiable patient uploads until the deployment has a documented consent basis, retention owner, incident process, backup/restore test, threat model, and approved medicine-dataset licensing. Configure request-size limits at the ingress/proxy as well as `MAX_UPLOAD_MB`; multipart bodies reach the server before application validation.

```bash
docker build -t simpliscribe .
docker run --rm -p 127.0.0.1:7860:7860 --env-file production.env simpliscribe
```

The image defaults to `APP_ENV=production` and fails closed without the complete production configuration above. Keep `production.env` outside the repository and secret manager values out of `.env.example`. Production session cookies require HTTPS: keep the container bound to loopback and place a TLS-terminating reverse proxy in front of `http://127.0.0.1:7860`; do not expose or browse the raw HTTP port directly. Use the local `uvicorn` workflow, not a remotely reachable Docker container, for unauthenticated development.

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
