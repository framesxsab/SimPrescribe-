# Delivery Phases

## Completed foundation

OCR, structured medication summaries, uncertainty labels, alternatives as reference candidates, professional review, PDF reports, authentication controls, audit events, and prescription history.

## Marketplace MVP

1. Native patient/pharmacy accounts and admin pharmacy approval.
2. Protected source retention plus versioned patient confirmation.
3. Pharmacy profiles, PIN matching, and exact-name inventory.
4. Quote, acceptance, stock recheck, COD pickup/delivery, and timelines.
5. Automated tests, browser smoke evidence, privacy/threat documentation, and migration verification. Live browser OCR now completes from the ignored writable runtime cache; structured results remain explicitly review-required.

## Deferred

Payments, refunds, logistics, maps, split fulfillment, external notifications, password reset/email verification, and any clinical decision functionality.

## Prescription core stabilization checkpoint

Persisted processing states, private source and OCR checkpoints, owner-scoped retry, patient review and saved edits, atomic confirmation, and the confirmation-gated pharmacy handoff are implemented. The patient upload hands off to a dedicated page that shows saved progress; refresh, retry, back/forward navigation, and confirmation were checked in the browser. The OCR cache is rooted to the project runtime directory, and the clean-browser OCR pass reached review. The development checkpoint passed automated tests, the synthetic golden gate, axe scans, keyboard checks, and mobile reflow checks. This does not establish clinical validation or production readiness; operational security and prospective clinical validation remain required before production use.
