# Dataset licensing and provenance inventory

This inventory describes the bundled medicine CSVs. The files stay in the repository. They are **reference lookup tables**, not a licensed formulary, not a prescribing source, and not clinical validation.

Operators must confirm upstream terms before redistribution, commercial hosting, or use with identifiable prescriptions. Code is MIT licensed; dataset files may have separate terms.

## Files that stay

| File | Role in SimpliScribe | Columns used (app) | Approximate size |
| --- | --- | --- | --- |
| `A_Z_medicines_dataset_of_India.csv` | Brand and composition lookup for same-composition candidates | `name`, `manufacturer_name`, `type`, `short_composition1`, `short_composition2` | ~254k rows |
| `all_medicine databased.csv` | Listed substitute brands for an unavailable medicine | `name`, `substitute0`–`substitute4` | ~248k rows |

Do not delete these CSVs as part of cleanup or unused-code jobs.

## Likely upstream sources (unverified against the exact bytes in this repo)

Column names and row counts are consistent with public Kaggle medicine tables commonly titled:

- [A-Z Medicine Dataset of India](https://www.kaggle.com/datasets/shudhanshusingh/az-medicine-dataset-of-india)
- [250k Medicines Usage, Side Effects and Substitutes](https://www.kaggle.com/datasets/shudhanshusingh/250k-medicines-usage-side-effects-and-substitutes) (often CC BY-SA 4.0 on Kaggle)

Related combined dumps appear under other Kaggle titles (some labeled MIT). **The license shown on a similar dataset page is not proof that this repo’s copies match that snapshot or license.** Treat provenance as incomplete until an operator records:

1. Exact download URL and date
2. Publisher / dataset version / checksum
3. License text that applies to that version
4. Whether share-alike or attribution obligations apply to derived apps

## Intended use and limits

- Local substitute and same-composition **reference candidates** only
- Displayed with human-verification labels; never recommendations
- Not a complete, current, or jurisdiction-correct drug database
- Side-effect and therapeutic-class columns in `all_medicine databased.csv` are not used for interaction decisioning, diagnosis, or reminders (those features are out of scope)

## Replacement path

When a licensed, versioned source with stable medicine identifiers is approved, add it beside these CSVs and switch lookups after documenting the new provenance. Do not silently overwrite the committed files.
