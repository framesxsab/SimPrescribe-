# SimpliScribe

SimpliScribe is a FastAPI application that extracts text from prescription images or PDFs and turns that OCR output into a structured medication summary.

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

1. Create a new Hugging Face Space with the Docker SDK.
2. Push this repository to the Space.
3. Add the environment variables from `.env.example` in the Space settings.
4. Expose port `7860`.

For an external hosted model, set `INFERENCE_PROVIDER=endpoint` and point `MODEL_API_URL` at your inference endpoint.