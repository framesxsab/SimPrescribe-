# Design Direction

## Principles

- Calm clinical workspace: slate surfaces, teal actions, amber safety warnings, and red destructive states.
- Put the prescription and verification status before commerce controls.
- Use plain language: requested, pharmacist quote, patient accepted, preparing, ready/out for delivery, fulfilled.
- Always distinguish OCR output, patient-entered changes, pharmacist notes, and informational generic candidates.

## Interface

- Continue the responsive Tailwind/Jinja system with rounded cards, strong headings, visible focus states, semantic labels, status text, and keyboard-operable native controls.
- Patient pages prioritize upload, review, pharmacy availability, and order timeline.
- The patient review page shows editable medicine fields, visible uncertainty, a link to the original, Save edits, and Confirm prescription. Upload opens a dedicated processing page that shows saved OCR and structuring stages without invented percentages. Failed processing offers a retry path.
- Pharmacy pages prioritize pending requests, original-prescription access, exact inventory, quote fields, and fulfillment actions.
