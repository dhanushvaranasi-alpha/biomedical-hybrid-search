# Usage: make setup | data | index | api | ui | test | eval
PY ?= python

setup:
	$(PY) -m pip install -e ".[dev]" && cd frontend && npm install

data:
	$(PY) scripts/download_data.py && $(PY) scripts/prepare_data.py

index:  # one-time, CPU: about 1 hour for 48,605 chunks on 2 cores (resumable)
	$(PY) scripts/build_indexes.py

api:
	$(PY) -m uvicorn app.main:app --app-dir backend --port 8000

ui:
	cd frontend && npm run dev

test:
	$(PY) -m pytest -q

eval:  # needs OPENROUTER_API_KEY in .env; about $3.5 for all five configs, results cached and capped by LLM_BUDGET_USD
	$(PY) -m eval.evaluate && $(PY) -m eval.latency && $(PY) -m eval.report

.PHONY: setup data index api ui test eval
