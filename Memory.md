# Project Memory

## Current product

SimpliScribe is a prescription-first FastAPI/Jinja marketplace MVP. It converts OCR output into structured medicine requirements, preserves patient edits as separate versions, matches approved pharmacies by Indian PIN code and exact normalized inventory names, and supports pharmacist-verified COD pickup/local-delivery orders.

## Fixed decisions

- No diagnosis, prescribing, automatic substitution, online payments, split orders, GPS distance, logistics, email, or SMS.
- Native patients and pharmacies use salted scrypt passwords; pharmacy access requires admin approval. Existing bootstrap/OIDC staff access remains.
- Generic/equivalent candidates are inquiry-only and never become order lines automatically.
- Prescription/source retention defaults to 30 days; order history defaults to 365 days.
- Preserve untracked `docs/uml/` work.

## Verification

Use the project virtual environment on Windows: `.venv\\Scripts\\python.exe -m pytest -q -p no:cacheprovider --basetemp=tmp\\pytest-run`. Run Ruff and Alembic upgrade checks alongside tests.

Release-readiness browser pass (2026-09-15): clean isolated database validated patient and pharmacy registration, pending-pharmacy 401, admin approval, inventory persistence, patient confirmation, PIN matching, one-pharmacy request, pharmacist source access, quote submission, patient COD acceptance, stock-backed fulfillment, order timelines/history, and unauthenticated source 401. Fixed missing favicon route, patient history status ambiguity, and Jinja `order.items` 500. Real PaddleOCR upload was attempted with `synthetic_prescription_dataset/images/prescription_0.png` but remains blocked on this workstation because Paddle tries to create `C:\Users\singa\\.cache\\paddle` and receives WinError 5; downstream validation used a realistic OCR-shaped seeded analysis and is not evidence that OCR completed in this environment.

OCR runtime follow-up (2026-09-16): installed versions are PaddleOCR 3.4.0, PaddlePaddle 3.3.0, and PaddleX 3.4.2. PaddleX honors `PADDLE_PDX_CACHE_HOME`; Paddle's hapi honors `PADDLE_HOME`, while Paddle's legacy dataset import expands the user profile directly. `OCR_CACHE_DIR` now write-tests a project/runtime directory; reader initialization temporarily routes `HOME`/`USERPROFILE`, `PADDLE_HOME`, and `PADDLE_PDX_CACHE_HOME` there and restores them afterward. Fresh browser OCR with the ignored configured cache completed at 91% confidence and persisted three structured medicine rows. The fallback parser now drops clinician/header/noise lines without medicine, form, strength, or timing evidence; the real image no longer surfaced `Dr. S Patel` or `Fox` as medicines.

Report layout follow-up (2026-09-16): PDF reports now compact ISO timestamps, left-align the hero title, show an explicit no-medicines state, start the verification appendix on a clean page, and keep each medication card/detail table together across pages. Rendered attached and multi-medication PDFs were visually inspected under `tmp/pdfs`; corrected deliverable is `output/pdf/pres_report_corrected.pdf`.
