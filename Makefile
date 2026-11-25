.PHONY: install test reproduce

install:
	pip install -e ".[dev]"

test:
	python -m pytest -q

reproduce:
	@test -n "$(INPUT)" || (echo "Set INPUT=/path/to/online_retail_II.xlsx" && exit 1)
	python scripts/run_survival.py --input-file "$(INPUT)" --out-dir reports
