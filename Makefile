PYTHON ?= python

.PHONY: run run-api test check-secrets docker-build docker-up docker-up-bifrost

run:
	$(PYTHON) -m streamlit run app.py --server.headless true --browser.gatherUsageStats false

run-api:
	$(PYTHON) -m uvicorn api:app --host 0.0.0.0 --port 8000

check-secrets:
	$(PYTHON) scripts/check_secrets.py

test: check-secrets
	$(PYTHON) -m unittest discover -s . -p "test_*.py" -v
	$(PYTHON) -m unittest discover -s lab/tests -p "test_*.py" -v

docker-build:
	docker build -t stale-detector:local .

docker-up:
	docker compose up --build

docker-up-bifrost:
	docker compose --profile bifrost up --build
