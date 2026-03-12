from simpliscribe.benchmark import load_cases
from simpliscribe.benchmark import run_case
from simpliscribe.benchmark import run_benchmark
from simpliscribe.benchmark import score_case


def test_score_case_counts_matching_fields():
    case = {
        "id": "case-1",
        "label": "Simple case",
        "raw_text": "Paracetamol 650 tab od 5 days",
        "expected_medications": [
            {
                "name": "Paracetamol",
                "type": "Tablet",
                "dosage": "650 mg",
                "frequency": "once daily",
                "duration": "5 days",
            }
        ],
    }
    actual = [
        {
            "name": "Paracetamol",
            "type": "Tablet",
            "dosage": "650 mg",
            "frequency": "once daily",
            "duration": "5 days",
        }
    ]

    result = score_case(case, actual)

    assert result.matched_fields == 5
    assert result.total_fields == 5
    assert result.score == 1.0


def test_run_benchmark_returns_summary(tmp_path):
    cases_path = tmp_path / "cases.json"
    cases_path.write_text(
        """
        [
          {
            "id": "case-1",
            "label": "Simple once-daily tablet",
            "raw_text": "Paracetamol 650 tab od 5 days",
            "expected_medications": [
              {
                "name": "Paracetamol",
                "type": "Tablet",
                "dosage": "650 mg",
                "frequency": "once daily",
                "duration": "5 days"
              }
            ]
          }
        ]
        """.strip(),
        encoding="utf-8",
    )

    cases = load_cases(cases_path)
    result = run_benchmark(cases)

    assert result["case_count"] == 1
    assert result["success_count"] == 1
    assert result["failure_count"] == 0
    assert result["cases"][0]["label"] == "Simple once-daily tablet"
    assert result["cases"][0]["actual"][0]["frequency"] == "once daily"


def test_run_benchmark_preserves_capsule_and_bedtime_in_mixed_case(tmp_path):
        cases_path = tmp_path / "cases.json"
        cases_path.write_text(
                """
                [
                    {
                        "id": "case-2",
                        "label": "Two-line mixed medicines",
                        "raw_text": "Amoxycillin 500 cap bd 5 days\\nCetirizine 10 tab hs 3 days",
                        "expected_medications": [
                            {
                                "name": "Amoxycillin",
                                "type": "Capsule",
                                "dosage": "500 mg",
                                "frequency": "twice daily",
                                "duration": "5 days"
                            },
                            {
                                "name": "Cetirizine",
                                "type": "Tablet",
                                "dosage": "10 mg",
                                "frequency": "at bedtime",
                                "duration": "3 days"
                            }
                        ]
                    }
                ]
                """.strip(),
                encoding="utf-8",
        )

        cases = load_cases(cases_path)
        result = run_benchmark(cases)

        assert result["average_score"] == 1.0
        assert result["cases"][0]["actual"][0]["type"] == "Capsule"
        assert result["cases"][0]["actual"][1]["frequency"] == "at bedtime"


def test_run_case_returns_failed_score_when_inference_errors(monkeypatch):
    case = {
        "id": "case-error",
        "label": "Broken case",
        "raw_text": "Broken OCR text",
        "expected_medications": [
            {"name": "Paracetamol", "type": "Tablet", "dosage": "650 mg", "frequency": "once daily", "duration": "5 days"}
        ],
    }

    def fake_structure_medications(_: str):
        raise RuntimeError("Synthetic benchmark failure")

    monkeypatch.setattr("simpliscribe.benchmark.structure_medications", fake_structure_medications)

    result = run_case(case, retries=0)

    assert result.score == 0.0
    assert result.error == "Synthetic benchmark failure"


def test_run_case_supports_file_based_ocr(monkeypatch, tmp_path):
    image_path = tmp_path / "sample.png"
    image_path.write_bytes(b"mock-image")

    case = {
        "id": "case-file",
        "label": "File-based case",
        "file_path": "sample.png",
        "expected_medications": [
            {
                "name": "Paracetamol",
                "type": "Tablet",
                "dosage": "650 mg",
                "frequency": "once daily",
                "duration": "5 days",
            }
        ],
    }

    monkeypatch.setattr("simpliscribe.benchmark.extract_ocr_text", lambda path: "Paracetamol 650 tab od 5 days")

    result = run_case(case, base_dir=tmp_path, retries=0)

    assert result.source_kind == "file"
    assert result.source_path == "sample.png"
    assert result.raw_text == "Paracetamol 650 tab od 5 days"
    assert result.actual[0]["frequency"] == "once daily"