# Dataset licensing and provenance inventory

This inventory records the exact upstream Kaggle records, source-file hashes, and separate terms for the two bundled medicine CSVs. The datasets and derived indexes are informational reference data, not a licensed formulary or clinical validation.

The application code is MIT licensed. The Kaggle source records license each dataset under CC BY-SA 4.0, separately from the code. The exact version-1 source-file bytes were downloaded and SHA-256 compared with the bundled Git LFS payloads on 2026-09-29.

## Verified redistribution terms

| Bundled file | Exact upstream file / dataset | Publisher and version | Size and SHA-256 match | License |
| --- | --- | --- | --- | --- |
| `A_Z_medicines_dataset_of_India.csv` | [A_Z_medicines_dataset_of_India.csv](https://www.kaggle.com/datasets/shudhanshusingh/az-medicine-dataset-of-india) | Shudhanshu Singh, *A-Z Medicine Dataset of India*, v1, 2022-11-17 | 32,061,801 bytes; `f89c1cdea39c615151201f15a99082e83c1aac0cd41a742d0f0c3a511c87a2d7` | CC BY-SA 4.0 |
| `all_medicine databased.csv` | Kaggle file `medicine_dataset.csv` from [250k Medicines Usage, Side Effects and Substitutes](https://www.kaggle.com/datasets/shudhanshusingh/250k-medicines-usage-side-effects-and-substitutes) | Shudhanshu Singh, v1, 2023-03-23 | 89,406,712 bytes; `d4eafe39da664bd96b66930b8630be697b4cc9fa0d03638d523f5d84343b5ae1` | CC BY-SA 4.0 |

The Kaggle version-1 file listings, names, sizes, creation dates, and license declarations were checked against the repository files on 2026-09-29. Each source file was downloaded temporarily and its contents hashed; both SHA-256 values above match the tracked Git LFS object exactly. The second CSV was renamed when bundled; its bytes were not changed.

## Attribution and derived indexes

For attribution, name Shudhanshu Singh, the exact dataset title, the linked Kaggle dataset page, and [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/). State that the second source file was renamed in this repository. Do not imply endorsement by the uploader or pharmaceutical companies.

The required SQLite lexicon index is derived from the A-Z dataset; the optional SQLite reference index is derived from the 250k dataset. If redistributed, these databases carry CC BY-SA 4.0 terms. Keep the MIT license for application code separate from the dataset/index notices in [THIRD_PARTY_NOTICES.md](../THIRD_PARTY_NOTICES.md).

The uploader's dataset descriptions say the records were compiled from pharmaceutical companies / manufacturers, but do not enumerate source URLs for each underlying fact. This audit verifies the uploader's exact v1 files and published license declaration; it does not establish separate trademark, privacy, or other rights not granted by that license. CC BY-SA 4.0 requires attribution and licensing shared adaptations under the same license; see the [official deed](https://creativecommons.org/licenses/by-sa/4.0/) and [legal code on database rights](https://creativecommons.org/licenses/by-sa/4.0/legalcode#s4).

Redistribution classification for both CSVs: **VERIFIED SHARE-ALIKE / SPECIAL TERMS**. The Kaggle uploader's CC BY-SA declaration is the basis for bundled redistribution; the MIT code license does not cover these datasets.

## Intended use and limits

- Local substitute and same-composition **reference candidates** only
- Displayed with human-verification labels; never recommendations
- Not a complete, current, or jurisdiction-correct drug database
- Side-effect and therapeutic-class columns in `all_medicine databased.csv` are not used for interaction decisioning, diagnosis, or reminders (those features are out of scope)

## Replacement path

When a licensed, versioned source with stable medicine identifiers is approved, add it beside these CSVs and switch lookups after documenting the new provenance. Do not silently overwrite the committed files.
