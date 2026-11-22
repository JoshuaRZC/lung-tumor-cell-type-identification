.PHONY: help download convert

help:
	@printf "Current configuration: configs/default.yml\n"
	@printf "make download  Download the raw data files\n"
	@printf "make convert   Extract selected samples and export Matrix Market files\n"

download:
	python scripts/download_data.py

convert:
	Rscript scripts/convert_raw_counts.R
