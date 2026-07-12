FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK=True
ENV PADDLEOCR_HOME=/app/.paddleocr

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
	libgl1 \
	libglib2.0-0 \
	libsm6 \
	libxext6 \
	libxrender1 \
	libgomp1 \
	&& rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

RUN mkdir -p /app/.paddleocr && python - <<'PY'
from paddleocr import PaddleOCR

for kwargs in (
    {'lang': 'en', 'device': 'cpu', 'use_textline_orientation': True, 'show_log': False},
    {'lang': 'en', 'device': 'cpu', 'use_textline_orientation': True},
    {'lang': 'en', 'use_angle_cls': True, 'use_gpu': False, 'show_log': False},
    {'lang': 'en', 'use_angle_cls': True, 'use_gpu': False}
):
    try:
        PaddleOCR(**kwargs)
        break
    except (TypeError, ValueError):
        pass

print("PaddleOCR models ready")
PY

COPY . .

RUN addgroup --system app && adduser --system --ingroup app app \
    && chown -R app:app /app

USER app

EXPOSE 7860

HEALTHCHECK --interval=30s --timeout=5s --start-period=90s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:7860/api/live', timeout=3)" || exit 1

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]
