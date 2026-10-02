.PHONY: test verify assets plots paper summaries all

test:
	python -m pytest -q

verify:
	python scripts/verify_paper_claims.py

assets:
	cd experiments && python make_paper_assets.py

plots:
	python scripts/make_result_plots.py

summaries:
	python scripts/collect_results.py

paper:
	cd paper && tectonic -X compile main.tex

all: test verify assets plots paper
