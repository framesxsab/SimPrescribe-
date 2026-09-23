# Engineering and Product Rules

- Never diagnose, prescribe, infer an interchangeable medicine, or calculate a dispensed quantity automatically.
- Preserve OCR evidence and label patient edits and pharmacist verification separately.
- Store the private source before OCR and OCR evidence before structuring. Only a confirmed owner-scoped prescription may reach pharmacy matching.
- Processing retries must claim a failed record atomically and must not create a second analysis. Patient draft saves and confirmation must preserve the original extraction.
- Stage logs use monotonic durations and analysis IDs; never log OCR text or medicine names.
- Match claimed pharmacy availability by exact normalized medicine name only.
- Authorize every health-data query by patient ownership, selected approved pharmacy, or explicit staff role; require CSRF on mutations.
- Keep prescription files private, generated-name, no-store, audited, and retention-bound.
- Keep OCR model/cache binaries in an ignored, configurable writable runtime directory; never commit them or place them under static assets.
- Validate OCR cache writability at startup and expose a clear configuration error instead of falling through to a generic OCR failure.
- Store money as integer paise and enforce order transitions server-side.
- Reuse FastAPI, Jinja, SQLAlchemy, and standard-library security primitives before adding dependencies.
- Preserve unrelated work and update tests and project documents with each phase.
