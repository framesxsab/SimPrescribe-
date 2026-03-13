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

COPY . .

EXPOSE 7860

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]