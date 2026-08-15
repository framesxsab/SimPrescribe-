# Contributing to SimpliScribe

Thank you for your interest in contributing to SimpliScribe! We welcome contributions from software engineers, clinical informatics specialists, and open-source advocates.

---

## Code of Conduct & Clinical Safety Principles

1. **Safety First**: SimpliScribe is a **human-in-the-loop review aid**, not an autonomous prescribing or clinical diagnosis system. Any feature or model change must preserve mandatory clinician review flags and fail-closed safety assertions.
2. **De-identification**: Never commit identifiable patient health information (PHI) or unconsented prescription images. All test fixtures must be synthetic or strictly de-identified with explicit consent.
3. **Golden Gate Rule**: All pull requests must maintain 100% test pass rate (`pytest`) and clean linter status (`ruff check .`). Changes to medication structuring logic must pass the versioned regression benchmark (`data/golden_cases.v1.json`).

---

## Local Development Setup

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/framesxsab/SimPrescribe-.git
   cd SimPrescribe-
   ```

2. **Create Virtual Environment & Install Dependencies**:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   pip install -r requirements.txt -r requirements-dev.txt
   ```

3. **Configure Environment**:
   ```bash
   cp .env.example .env
   ```

4. **Run the Development Server**:
   ```bash
   uvicorn app:app --reload
   ```
   Navigate to `http://127.0.0.1:8000`.

---

## Quality Checks & Testing

Before submitting a Pull Request, run the full validation suite:

```bash
# 1. Run unit and integration tests
pytest -q

# 2. Run linter
ruff check .

# 3. Run golden dataset regression benchmark
python -m simpliscribe.benchmark --cases data/golden_cases.v1.json

# 4. (Optional) Rebuild vector embeddings index
python scripts/build_embeddings.py --benchmark
```

---

## Adding Clinician Golden Cases

When adding de-identified, clinician-adjudicated benchmark cases:
* **Target File**: Append new cases **only** to `data/golden_cases.clinician.v1.json`.
* **Never rewrite** `data/golden_cases.v1.json` (the versioned synthetic baseline).
* Ensure all cases conform to schema `1.0` and include unique IDs and ground truth medication arrays.

---

## Submitting Pull Requests

1. Create a feature branch: `git checkout -b feat/your-feature-name`
2. Commit with conventional commit messages: `feat: ...`, `fix: ...`, `docs: ...`, `test: ...`
3. Push to your fork and submit a PR against `main`.
4. Ensure CI tests and ruff linting pass on GitHub Actions.
