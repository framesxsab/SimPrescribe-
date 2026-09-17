# Architecture

## Flow

FastAPI and Jinja serve role-aware pages and JSON APIs. PaddleOCR and the existing inference pipeline produce an owner-scoped analysis. A patient confirmation creates a versioned edited view without removing the OCR snapshot. Approved pharmacies are matched by exact or configured serviceable PIN and exact normalized inventory names. Orders snapshot prescription lines and advance through audited quote and fulfillment states.

## Data and storage

- SQLAlchemy tables: analyses, audit events, vector cache, users, pharmacies, inventory, prescription files, orders, order items, and order events.
- SQLite remains the local default; PostgreSQL is required in production. Alembic owns production migrations.
- Source prescriptions use generated filenames outside static serving. Database metadata controls authorization and 30-day expiry.
- OCR runtime data uses `OCR_CACHE_DIR` (default `tmp/ocr-cache`); Paddle 3.3 uses `PADDLE_HOME` and PaddleX 3.4 uses `PADDLE_PDX_CACHE_HOME`. Initialization write-tests the configured directory and temporarily routes legacy home expansion there without changing the persistent user profile.
- Order prices are integer paise; completed order history defaults to 365 days.

## Main modules

- `simpliscribe/main.py`: routes, sessions, role-aware pages, and API boundary.
- `simpliscribe/web.py`: OCR analysis, patient/professional review, reports, and protected sources.
- `simpliscribe/marketplace.py`: matching, inventory, quoting, stock checks, and order transitions.
- `simpliscribe/storage.py`: relational schema and persistence primitives.
