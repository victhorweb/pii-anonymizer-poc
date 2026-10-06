PORT ?= 8765
HOST ?= 127.0.0.1

.PHONY: install run test

install:
	python3 -m venv .venv
	.venv/bin/pip install --upgrade pip
	.venv/bin/pip install -r requirements.txt

run:
	PORT=$(PORT) HOST=$(HOST) ./run.sh

test:
	.venv/bin/python -m pytest -q
