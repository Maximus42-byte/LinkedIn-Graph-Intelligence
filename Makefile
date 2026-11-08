PYTHON := .venv/bin/python

.PHONY: install fixtures run test clean

install:
	@python3 -c 'import sys; version = sys.version_info; sys.exit("Python 3.11 or newer is required; found %s.%s." % version[:2] if version < (3, 11) else 0)'
	python3 -m venv .venv
	$(PYTHON) -m pip install -r requirements.txt

fixtures:
	PYTHONPATH=src $(PYTHON) -m lgi.fixtures_gen.generator --n 10

run:
	PYTHONPATH=src $(PYTHON) -m lgi.cli run --source fixture

test:
	PYTHONPATH=src $(PYTHON) -m pytest -q

clean:
	rm -rf data/profiles.json data/profiles.csv output
