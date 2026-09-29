# Changelog

This changelog records user-visible release changes. Dates use ISO 8601.

## [Unreleased]

## [0.2.0-rc.1] - 2026-09-28

### Added

- Persisted prescription processing and retry, patient review versions, confirmation, and a PIN-based pharmacy marketplace.
- Fingerprinted SQLite indexes for required medicine matching and optional reference enrichment.
- Patient-facing processing, review, details, pharmacy, and order workflows with accessibility checks.
- Windows and WSL/Linux setup guidance, strict production migration steps, and PostgreSQL backup/restore verification.

### Changed

- OCR reader initialization runs as background warmup; readiness reports initializing and ready without blocking liveness.
- Optional reference detail lookups use targeted SQLite access instead of loading the full CSV in a request.
- Docker and CI fetch Git LFS datasets and build required indexes from them.

### Fixed

- Pharmacy/order transitions validate state and stock transactionally to protect against competing requests.
- Production startup now requires an explicitly migrated database; demo schema bootstrap remains for local development and tests.
- Windows runtime setup pins the validated Python 3.11 OCR stack and keeps its venv separate from WSL/Linux.

### Security

- Added private GitHub vulnerability reporting and clarified that prescription/OCR results require human review.
- Tightened production secret requirements and documented source-data licensing separately from the MIT code license.
