PYTHON := .venv/bin/python
UV := uv

.PHONY: install fixtures run test clean

install:
	$(UV) venv --python 3.12 .venv
	$(UV) pip install --python $(PYTHON) -r requirements.txt

fixtures:
	PYTHONPATH=src $(PYTHON) -m lgi.fixtures_gen.generator --n 10

run:
	PYTHONPATH=src $(PYTHON) -m lgi.cli run --source fixture

test:
	PYTHONPATH=src $(PYTHON) -m pytest -q

clean:
	rm -rf data/profiles.json data/profiles.csv output
