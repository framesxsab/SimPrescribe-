# Roadmap: SimpliScribe

## Overview

SimpliScribe provides prescription OCR structuring, medicine reference enrichment, and pharmacist-verified fulfillment. This roadmap captures the completed foundation and marketplace MVP, followed by gap closure phases to ensure a seamless, non-cluttered user flow from prescription scan to order delivery, running from a single unified terminal.

## Phases

- [x] **Phase 1: Accounts & Roles** - Native patient/pharmacy accounts and admin approvals.
- [x] **Phase 2: Source Retention & Verification** - Protected file retention with versioned patient confirmation.
- [x] **Phase 3: Profiles & Inventory** - Pharmacy profiles, PIN matching, and exact-name inventory.
- [x] **Phase 4: Quotes & Fulfillment** - Quote submission, stock checks, COD acceptance, and order lifecycle.
- [x] **Phase 5: Release Readiness** - Automated test suite verification (149 tests).
- [ ] **Phase 6: Unified Runtime & Stability (Gap Closure)** - Single-terminal startup with seed pharmacies, Windows file-lock safety, and offline OCR isolation.
- [ ] **Phase 7: End-to-End User Flow & Navigation (Gap Closure)** - Seamless transitions: Dashboard CTA to Details, guest PIN search with guided auth, and global breadcrumbs.
- [ ] **Phase 8: Calm Clinical UI & Prescription Verification (Gap Closure)** - 4-step visual flow stepper, clear review guidance, and side-by-side original prescription document viewer.

---

## Phase Details

### Phase 1: Accounts & Roles
**Goal**: Native patient/pharmacy accounts and admin approvals.
**Requirements**: AUTH-01
**Success Criteria**: Accounts register and authenticate; pharmacy requires admin approval.

### Phase 2: Source Retention & Verification
**Goal**: Protected file retention with versioned patient confirmation.
**Requirements**: RET-01
**Success Criteria**: Original files stored securely and patient reviews versioned.

### Phase 3: Profiles & Inventory
**Goal**: Pharmacy profiles, PIN matching, and exact-name inventory.
**Requirements**: MKT-01
**Success Criteria**: Exact medicine inventory matched against patient PINs.

### Phase 4: Quotes & Fulfillment
**Goal**: Quote submission, stock checks, COD acceptance, and order lifecycle.
**Requirements**: ORD-01
**Success Criteria**: Orders transition through quote, acceptance, and fulfillment.

### Phase 5: Release Readiness
**Goal**: Automated test suite verification (149 tests).
**Requirements**: TEST-01
**Success Criteria**: All tests pass cleanly without errors.

### Phase 6: Unified Runtime & Stability (Gap Closure)
**Goal**: The entire application starts and runs cleanly in a single terminal with pre-seeded test pharmacies, zero remote model check freezes, and zero Windows file-lock crashes.
**Requirements**: SYS-01, SYS-02
**Gap Closure**: Closes audit gaps SYS-01, SYS-02
**Success Criteria**:
1. Running `python app.py` starts the server on port 8000 with pre-seeded approved pharmacies serving test PINs (560001, 110001, 400001).
2. PaddleOCR runs locally without remote network timeouts via `PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK`.
3. Prescription uploads and deletions never raise Windows `PermissionError` [WinError 32].

### Phase 7: End-to-End User Flow & Navigation (Gap Closure)
**Goal**: Complete seamless transition across all application flows so users are never left stranded.
**Requirements**: FLOW-01, FLOW-02, FLOW-03
**Gap Closure**: Closes audit gaps FLOW-01, FLOW-02, FLOW-03
**Success Criteria**:
1. Scanning a prescription on Dashboard renders a prominent Next Step Action Card linking directly to `/details/{id}#medicine-marketplace`.
2. Unauthenticated visitors can enter a PIN code to view matching approved pharmacies and their stock availability.
3. Clicking "Request pharmacist quote" as a guest prompts them to sign in or register with a seamless return URL.
4. Header features persistent Sign In / Register buttons for guests, and Details page includes breadcrumb back-navigation.

### Phase 8: Calm Clinical UI & Prescription Verification (Gap Closure)
**Goal**: Replace cluttered AI debug strings with a calm, professional clinical UI and visual verification.
**Requirements**: UI-01, UI-02
**Gap Closure**: Closes audit gaps UI-01, UI-02
**Success Criteria**:
1. Dashboard features a clean 4-step visual progress stepper (Upload → Verify → Match Pharmacies → Order).
2. Details page features a toggleable side-by-side Original Prescription Document viewer pulling from `/api/analyses/{id}/source`.
3. Technical debug jargon ("labeled degraded output", "cached vector match") is replaced by calm human review notices.
