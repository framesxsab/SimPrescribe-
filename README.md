---
title: SimpliScribe
sdk: docker
app_port: 7860
pinned: false
---

# SimpliScribe

SimpliScribe is a FastAPI application that simplifies prescription reading by extracting text from prescription images or PDFs and turning that OCR output into a structured medication summary.

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

Open `http://127.0.0.1:8000`.

## Testing

```bash
pytest
```

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
