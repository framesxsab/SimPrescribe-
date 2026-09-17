# Requirements: SimpliScribe

**Defined:** 2026-09-16
**Core Value:** Turn scanned prescriptions into verified structured medication orders and connect patients with approved local pharmacies in a single cohesive, seamless workflow running from a single unified server.

## Milestone Requirements (Gap Closure)

### User Flow & Navigation

- [ ] **FLOW-01**: Dashboard (`/`) prescription scan provides a clear, prominent call-to-action ("Review & Find Pharmacies") that deep-links directly into `/details/{id}#medicine-marketplace`.
- [ ] **FLOW-02**: Details page (`/details/{id}`) allows unauthenticated/guest users to preview PIN-based local pharmacies, providing frictionless redirect to `/login` or `/register` when requesting quotes.
- [ ] **FLOW-03**: Global navigation provides clear breadcrumbs (`Dashboard > Prescriptions > Details > Fulfillment`), return buttons, and persistent header Sign In / Register buttons for guests.

### Grounded Clinical UI

- [ ] **UI-01**: Top of Dashboard features a clear 4-step visual progress stepper (1. Upload & Scan → 2. Review Medications → 3. Select Pharmacy → 4. Order Quote) with clear human review banners instead of raw AI debug strings.
- [ ] **UI-02**: Details page provides an interactive, collapsible side-by-side Original Prescription Document viewer pulling from `/api/analyses/{id}/source` for visual verification.

### Unified Runtime & Single-Terminal Execution

- [ ] **SYS-01**: The entire application starts and runs cleanly with a single command (`python app.py` or `uvicorn simpliscribe.main:app`), auto-initializing SQLite schemas and seeding approved test pharmacies for sample PINs (560001, 110001, 400001).
- [ ] **SYS-02**: Offline PaddleOCR isolation (`PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK = "True"`) and Windows file-locking protection (`safe_unlink`) prevent upload freezes and HTTP 500 crashes.

## Traceability

| Requirement | Phase | Status |
|-------------|-------|--------|
| SYS-01 | Phase 6 | Pending |
| SYS-02 | Phase 6 | Pending |
| FLOW-01 | Phase 7 | Pending |
| FLOW-02 | Phase 7 | Pending |
| FLOW-03 | Phase 7 | Pending |
| UI-01 | Phase 8 | Pending |
| UI-02 | Phase 8 | Pending |
