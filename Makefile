#* Variables
SHELL := /usr/bin/env bash
PYTHON := python

#* Poetry
.PHONY: poetry-download
poetry-download:
	curl -sSL https://raw.githubusercontent.com/python-poetry/poetry/master/install-poetry.py | POETRY_HOME=$(HOME)/.poetry $(PYTHON) -

.PHONY: poetry-remove
poetry-remove:
	curl -sSL https://raw.githubusercontent.com/python-poetry/poetry/master/install-poetry.py | POETRY_HOME=$(HOME)/.poetry $(PYTHON) - --uninstall

#* Installation
.PHONY: install
install:
	poetry install

.PHONY: pre-commit-install
pre-commit-install:
	poetry run pre-commit install

.PHONY: pre-commit-run
pre-commit-run:
	poetry run pre-commit run --all-files

.PHONY: black
black:
	poetry run black .

.PHONY: isort
isort:
	poetry run isort .

.PHONY: tests
tests:
	poetry run pytest

.PHONY: coverage
coverage:
	poetry run coverage run -m pytest
	poetry run coverage report -m

.PHONY: coverage-html
coverage-html:
	poetry run coverage run -m pytest
	poetry run coverage html
