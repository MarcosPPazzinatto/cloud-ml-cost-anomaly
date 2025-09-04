.PHONY: test lint run

install:
	pip install -r requirements.txt

test:
	pytest -q

lint:
	flake8 .

run-local-mlflow:
	docker compose -f ops/docker/mlflow.compose.yml up -d
