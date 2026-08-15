# SimpliScribe PowerShell Helper Script
param (
    [Parameter(Position = 0)]
    [ValidateSet("test", "lint", "coverage", "benchmark", "build-index", "run", "clean")]
    [string]$Task = "test"
)

$VenvPython = Join-Path $PSScriptRoot "..\.venv\Scripts\python.exe"
$VenvPytest = Join-Path $PSScriptRoot "..\.venv\Scripts\pytest.exe"
$VenvRuff = Join-Path $PSScriptRoot "..\.venv\Scripts\ruff.exe"
$VenvUvicorn = Join-Path $PSScriptRoot "..\.venv\Scripts\uvicorn.exe"

if (-not (Test-Path $VenvPython)) {
    $VenvPython = "python"
    $VenvPytest = "pytest"
    $VenvRuff = "ruff"
    $VenvUvicorn = "uvicorn"
}

switch ($Task) {
    "test" {
        & $VenvPytest -q
    }
    "lint" {
        & $VenvRuff check .
    }
    "coverage" {
        & $VenvPytest --cov=simpliscribe --cov-report=term-missing
    }
    "benchmark" {
        & $VenvPython -m simpliscribe.benchmark --cases data/golden_cases.v1.json
    }
    "build-index" {
        & $VenvPython scripts/build_embeddings.py --benchmark
    }
    "run" {
        & $VenvUvicorn app:app --host 127.0.0.1 --port 8000 --reload
    }
    "clean" {
        Get-ChildItem -Path . -Include __pycache__, .pytest_cache, .ruff_cache -Recurse -Force | Remove-Item -Recurse -Force
        Write-Host "Caches cleaned."
    }
}
