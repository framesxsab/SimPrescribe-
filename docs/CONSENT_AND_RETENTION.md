# Consent and retention (operations)

SimpliScribe is a prescription **review aid**. It must not be operated as a diagnosis, prescribing, reminder, or drug-interaction decision system.

This note is for operators. It is not a privacy policy, DPIA, or legal advice.

## Consent (application)

- Upload processing requires an explicit checkbox: “I consent to processing this prescription for OCR and medication review.” (`consent` form field; `web.analyze` rejects missing consent with HTTP 400).
- The UI states that results require confirmation against the original document.
- The processing copy is deleted after OCR. A generated-name copy is retained in private, non-static prescription storage for pharmacist verification and is accessible only to the patient, the selected approved pharmacy, and authorized admin/reviewer roles.
- Do not enable identifiable patient uploads until a documented lawful basis (or equivalent) exists for the deployment’s jurisdiction.

## What the product stores

- Structured analysis JSON (OCR text, extracted fields, pipeline metadata, review versions) in PostgreSQL/SQLite `analyses.payload`
- Native account profiles, approved pharmacy profiles, exact-name inventory, order snapshots, quotes, and order timelines
- Protected prescription files with hashes and expiry metadata in `prescription_files`
- Owner-scoped audit events (`login_succeeded`, `analysis_created`, review events) without raw images
- Session cookies (signed, HTTP-only in production)

It does not implement consultation/diagnosis records, clinical decisioning, or automatic substitution.

## Retention

- `RETENTION_DAYS` (default 30, minimum 1) sets `analyses.expires_at` at insert time (`storage._insert_record`).
- Protected prescription files use the same 30-day default. `ORDER_RETENTION_DAYS` defaults to 365 for minimal order and event history.
- `purge_expired()` deletes rows whose `expires_at` is in the past; it runs when history is loaded.
- Reducing `RETENTION_DAYS` does not rewrite existing rows. Expired rows are removed on the next purge.
- Audit events are not expired by this setting; treat them as operator-controlled records and purge them in the approved operations process if required.
- Backups follow the same sensitivity as the live database. Recovery drills must use a disposable restore database (see `PRODUCTION_RECOVERY.md`).

## Roles and shared deployments

- Native patient and pharmacy credentials use salted `scrypt`; pharmacies remain pending until admin approval. Prefer OpenID Connect for shared staff reviewer access (`OIDC_*`).
- Keep `ADMIN_EMAIL` / `ADMIN_PASSWORD` as an emergency bootstrap account only. When OIDC is configured and a bootstrap password is set, the login page shows organization sign-in first.
- `owner_id` isolates analyses per signed-in identity. This is not a multi-tenant product beyond that isolation.

## Operator checklist before identifiable data

1. Named retention owner and deletion process
2. Consent / lawful-basis text approved for the deployment
3. Incident process and backup/restore drill recorded
4. Threat model reviewed (`simpliscribe-threat-model.md`)
5. Dataset licensing confirmed (`DATASET_PROVENANCE.md`)
6. Medicine-dataset alternatives: keep `ALTERNATIVES_ENABLED=false` unless off-box name lookup is approved
