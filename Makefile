.PHONY: help install dev test lint coverage benchmark build-index run docker-build docker-prod clean

help:
	@echo "SimpliScribe Development & Production Commands:"
	@echo "  make install       Install core dependencies"
	@echo "  make dev           Install dev dependencies (pytest, ruff, etc.)"
	@echo "  make test          Run pytest test suite"
	@echo "  make lint          Run ruff linter check"
	@echo "  make coverage      Run pytest with coverage report"
	@echo "  make benchmark     Run clinical golden regression benchmark"
	@echo "  make build-index   Build/update dense prescription embeddings vector index"
	@echo "  make run           Start development server on port 8000"
	@echo "  make docker-build  Build Docker image"
	@echo "  make docker-prod   Start production stack with Docker Compose"
	@echo "  make clean         Clean temporary caches and artifacts"

install:
	pip install -r requirements.txt

dev:
	pip install -r requirements.txt -r requirements-dev.txt

test:
	pytest -q

lint:
	ruff check .

coverage:
	pytest --cov=simpliscribe --cov-report=term-missing

benchmark:
	python -m simpliscribe.benchmark --cases data/golden_cases.v1.json

build-index:
	python scripts/build_embeddings.py --benchmark

run:
	uvicorn app:app --host 127.0.0.1 --port 8000 --reload

docker-build:
	docker build -t simpliscribe:latest .

docker-prod:
	docker compose -f docker-compose.prod.yml up -d

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
	find . -type d -name ".ruff_cache" -exec rm -rf {} +
