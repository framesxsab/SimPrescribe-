---
title: SimpliScribe
sdk: docker
app_port: 7860
pinned: false
---

# SimpliScribe

SimpliScribe is a FastAPI application that simplifies prescription reading by extracting text from prescription images or PDFs and turning that OCR output into a structured medication summary.

## Open-source local stack

- OCR runs locally with PaddleOCR.
- Medication structuring can run with `INFERENCE_PROVIDER=fallback` or a self-hosted endpoint via `INFERENCE_PROVIDER=endpoint`.
- No external paid API is required if you keep OCR local and point the endpoint mode at your own deployed model.

## Local model server

This repo includes a separate local model server that you can run on your own laptop and point the main app to.

Install the optional model-serving dependencies:

```bash
pip install -r requirements-local-model.txt
```

Start the model server on port `8001`:

```bash
uvicorn simpliscribe.local_model_server:app --host 127.0.0.1 --port 8001
```

Then point the main app at that local endpoint:

```env
INFERENCE_PROVIDER=endpoint
MODEL_API_URL=http://127.0.0.1:8001/extract
LOCAL_MODEL_ID=Qwen/Qwen2.5-1.5B-Instruct
LOCAL_MODEL_DEVICE=auto
LOCAL_MODEL_TEMPERATURE=0.1
LOCAL_MODEL_MAX_NEW_TOKENS=256
```

The default local model is intentionally small enough to be more realistic on consumer hardware. If you have a stronger GPU, you can raise `LOCAL_MODEL_ID` to a larger open model.

For a 6 GB GPU, `Qwen/Qwen2.5-1.5B-Instruct` is the recommended default starting point before trying larger models.
The first request can take a few minutes because model weights may need to download and load into memory, so the default endpoint timeout is intentionally longer for local use.

## Runtime options

- `INFERENCE_PROVIDER=fallback`
  Uses a local heuristic fallback and does not require external model credentials.
- `INFERENCE_PROVIDER=huggingface`
  Uses the Hugging Face Inference API with `HUGGINGFACEHUB_API_TOKEN` and `HF_CHAT_MODEL`.
- `INFERENCE_PROVIDER=endpoint`
  Sends OCR text to a compatible HTTP endpoint using `MODEL_API_URL` and optional `MODEL_API_KEY`.

## Local development

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
copy .env.example .env
uvicorn app:app --reload
```

Recommended local environment variables for a fully self-hosted setup:

```env
INFERENCE_PROVIDER=endpoint
MODEL_API_URL=http://127.0.0.1:8001/extract
OCR_LANGUAGE=en
OCR_USE_GPU=false
```

If you do not want to run the local model server yet, keep `INFERENCE_PROVIDER=fallback` and the app will stay fully local with rule-based extraction only.

Open `http://127.0.0.1:8000`.

## Testing

```bash
pytest
```

## Benchmarking

You can benchmark extraction quality locally against curated OCR text cases or full image/PDF cases.

Run the benchmark with the current provider configuration:

```bash
python -m simpliscribe.benchmark --cases data/benchmark_cases.sample.json
```

This writes a JSON report to `data/benchmark_runs/latest.json` and prints a summary score in the terminal.

Case files support either:

- `raw_text`: benchmark only the structuring stage
- `file_path`: benchmark the full OCR + structuring pipeline

Example file-based case shape:

```json
[
  {
    "id": "scan-1",
    "label": "Prescription image",
    "file_path": "../uploads/prescription.png",
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
```

Recommended workflow:

1. Add real OCR text samples to `data/benchmark_cases.sample.json` or a separate JSON file.
2. Add file-based cases when you want to measure OCR and structuring together.
3. Run once with `INFERENCE_PROVIDER=fallback`.
4. Run again with `INFERENCE_PROVIDER=endpoint` and your local model server.
5. Compare the saved benchmark reports before changing prompts or models.

## Docker deployment

```bash
docker build -t simpliscribe .
docker run -p 7860:7860 --env-file .env simpliscribe
```

## Hugging Face Spaces

This project is ready for a Docker Space.

Recommended Space path: `fxsab/simpliscribe`

1. In your Hugging Face account, create a new Space under `fxsab`.
2. Choose `Docker` as the SDK.
3. Name the Space `simpliscribe`.
4. Push this repository to `https://huggingface.co/spaces/fxsab/simpliscribe`.
5. Add the environment variables from `.env.example` in the Space settings.
6. Keep `INFERENCE_PROVIDER=fallback` if you want zero model subscription cost.

Example git remote setup:

```bash
git remote add space https://huggingface.co/spaces/fxsab/simpliscribe
git push space master
```

Suggested Space variables:

```env
APP_NAME=SimpliScribe
APP_ENV=production
INFERENCE_PROVIDER=fallback
HUGGINGFACEHUB_API_TOKEN=
HF_CHAT_MODEL=Qwen/Qwen2.5-7B-Instruct
MODEL_API_URL=
MODEL_API_KEY=
MAX_UPLOAD_MB=10
REQUEST_TIMEOUT_SECONDS=60
```

Expected behavior on Spaces:

- The app is publicly viewable through the Space URL.
- OCR still runs inside the Space container.
- No paid model API is required when `INFERENCE_PROVIDER=fallback`.
- Performance depends on the free CPU resources available to the Space.

For an external hosted model, set `INFERENCE_PROVIDER=endpoint` and point `MODEL_API_URL` at your inference endpoint.
