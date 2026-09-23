# Architecture

## Flow

FastAPI and Jinja serve role-aware pages and JSON APIs. PaddleOCR and the existing inference pipeline produce an owner-scoped analysis. A patient confirmation creates a versioned edited view without removing the OCR snapshot. Approved pharmacies are matched by exact or configured serviceable PIN and exact normalized inventory names. Orders snapshot prescription lines and advance through audited quote and fulfillment states.

The patient upload path is: validate upload -> persist `uploaded` analysis and private source -> redirect to the processing page -> claim `processing` -> OCR -> persist OCR text/lines -> structure medicines -> persist `review_required` -> patient draft/edit -> `confirmed`. The processing page polls an owner-scoped status API for actual saved stages; refresh reads the same analysis without restarting it. Failed processing retains the source and any completed OCR, and retry repeats OCR only if OCR text is absent. The required lexicon loads from a fingerprinted SQLite index when present and current; the authoritative CSV builder remains available through `python -m simpliscribe.build_lexicon_index`. A missing or stale index uses CSV construction; an existing stale index logs a warning. The generated SQLite database is a rebuildable runtime artifact. Substitutes, uses, side effects, and composition peer indexes load only when reference enrichment or report/details rendering requests them. Off-box alternative lookup is skipped during structuring. PDF generation remains on request. `/api/analyze` remains a synchronous compatibility route.

OCR reader initialization runs once in a background lifespan task for non-test processes. `/api/live` remains a cheap liveness check; `/api/health` reports `ocr_state`, `ocr_ready`, and a safe OCR error code. Upload processing shares the same reader lock while warmup is running, so an early upload waits on the existing initialization and keeps its persisted analysis.

## Data and storage

- SQLAlchemy tables: analyses, audit events, vector cache, users, pharmacies, inventory, prescription files, orders, order items, and order events.
- SQLite remains the local default; PostgreSQL is required in production. Alembic owns production migrations.
- Source prescriptions use generated filenames outside static serving. Database metadata controls authorization and 30-day expiry.
- Prescription processing state, stage, OCR lines, original medicine snapshot, patient draft, and version history live in the existing JSON analysis payload. No table schema changed.
- OCR runtime data uses `OCR_CACHE_DIR` (default `tmp/ocr-cache`); relative cache settings resolve from the repository root, and absolute overrides are preserved. PaddleOCR import and initialization run with Paddle 3.3 `PADDLE_HOME`, PaddleX 3.4 `PADDLE_PDX_CACHE_HOME`, and legacy home expansion pointed at the configured cache, then the process environment is restored. Startup write-tests the configured directories.
- `npm run test:a11y` runs axe-core through Playwright against the patient pages. Set a base URL and local audit-account credentials; optional owned analysis/order IDs add processing, review, details, and order-detail scans.
- Order prices are integer paise; completed order history defaults to 365 days.

## Main modules

- `simpliscribe/main.py`: routes, sessions, role-aware pages, and API boundary.
- `simpliscribe/web.py`: OCR analysis, patient/professional review, reports, and protected sources.
- `simpliscribe/marketplace.py`: matching, inventory, quoting, stock checks, and order transitions.
- `simpliscribe/storage.py`: relational schema and persistence primitives.
