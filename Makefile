default: run

################################################################################
# Application run

run:
	.\.venv\Scripts\streamlit run .\src\app.py


################################################################################
# Linting, formatting, typing, security checks, coverage

test:
	python -m pytest --log-cli-level info -p no:warnings -v -s ./tests

format:
	ruff format --diff ./src
	ruff check --select I --fix --show-fixes ./src

type:
	python -m mypy --no-implicit-reexport --ignore-missing-imports --no-namespace-packages ./src

lint:
	ruff check --diff ./src

secu:
	python -m bandit -rq ./src

ci: lint format type secu

