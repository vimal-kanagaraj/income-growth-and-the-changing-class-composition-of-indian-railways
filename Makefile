PYTHON ?= python3
VENV   := .venv
BIN    := $(VENV)/bin
export PYTHONPATH := src

.PHONY: install run test lint format clean

$(BIN)/python:
	$(PYTHON) -m venv $(VENV)

install: $(BIN)/python
	$(BIN)/pip install -e ".[dev]"

run:
	$(BIN)/python -m railclass --output-dir outputs

test:
	$(BIN)/pytest

lint:
	$(BIN)/ruff check src tests
	$(BIN)/ruff format --check src tests

format:
	$(BIN)/ruff check --fix src tests
	$(BIN)/ruff format src tests

clean:
	rm -rf outputs .pytest_cache .ruff_cache
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
