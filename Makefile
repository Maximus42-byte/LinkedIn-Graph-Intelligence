PYTHON := .venv/bin/python
UV := uv
PYTHON_VERSION ?= 3.12

.PHONY: install fixtures run test clean

install:
	test -x $(PYTHON) || $(UV) venv --offline --python $(PYTHON_VERSION) .venv
	$(UV) pip install --offline --python $(PYTHON) -r requirements.txt

fixtures:
	PYTHONPATH=src $(PYTHON) -m lgi.fixtures_gen.generator --n 10

run:
	PYTHONPATH=src $(PYTHON) -m lgi.cli run --source fixture

test:
	PYTHONPATH=src $(PYTHON) -m pytest -q

clean:
	rm -rf data/profiles.json data/profiles.csv output
