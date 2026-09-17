# SimpliScribe

## What This Is

SimpliScribe converts difficult-to-read prescription images and PDFs into structured medication summaries and connects patients with approved local pharmacies for medicine availability, pharmacist verification, and cash-on-delivery orders. It features an integrated local OCR engine, Indian medicine datasets, PIN-based pharmacy discovery, patient confirmation versioning, and pharmacist quote-and-fulfill order lifecycles.

## Core Value

Turn scanned prescriptions into verified structured medication orders and connect patients with approved local pharmacies in a single cohesive, seamless workflow running from a single unified server.

## Requirements

### Validated

- [x] OCR text extraction with confidence scoring and rotation support via PaddleOCR
- [x] Fallback heuristic parser extracting medicine name, dosage, frequency, and duration
- [x] Local Indian medicine dataset enrichment (composition, manufacturer, alternatives)
- [x] Professional reviewer confirmation and PDF report export
- [x] Native scrypt-based patient and pharmacy authentication
- [x] Admin approval workflow for pharmacies
- [x] PIN code matching between patients and pharmacies
- [x] Order quoting, COD acceptance, stock recheck, and order status transitions

### Active (Gap Closure Scope)

- [ ] **FLOW-01**: Seamless transition from prescription upload & scan on Dashboard directly to Review & Details page (`/details/{id}`)
- [ ] **FLOW-02**: Unauthenticated guest access to PIN-based pharmacy matching with clear login/register prompts before quote placement
- [ ] **FLOW-03**: Cohesive breadcrumb navigation, return links, and header Sign-In/Register action buttons across all views
- [ ] **UI-01**: Clear 4-step visual flow stepper (Upload → Verify → Match Pharmacies → Order) replacing AI-jargon clutter
- [ ] **UI-02**: Collapsible side-by-side original prescription document viewer on Details page
- [ ] **SYS-01**: Unified single-terminal command (`python app.py`) with pre-seeded sample pharmacies and zero network freeze
- [ ] **SYS-02**: Windows file-lock safety (`safe_unlink`) and PaddleX offline isolation preventing 500 crashes and upload hangs

### Out of Scope

- Clinical decision support, diagnosis, or prescribing
- Automatic generic drug substitution without pharmacist consent
- Online payment processing, refunds, or payment gateways (COD only)
- Real-time GPS mapping or fleet logistics tracking

## Context

- Operating System: Windows 11 with PowerShell
- Framework: FastAPI + Jinja2 + Tailwind CSS + SQLite
- OCR Engine: PaddleOCR 3.4.0 + PaddleX 3.4.2 running locally
- Python Environment: `.venv\Scripts\python.exe` with Pytest test suite (149 tests)

## Key Decisions

- Run as a single unified service on `http://127.0.0.1:8000` via `python app.py`
- Retain original prescriptions under protected storage (`data/prescriptions`) with 30-day retention
- Require explicit human verification for all extracted medication fields before pharmacy ordering
