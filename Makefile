.PHONY: help download convert explore preprocess representation

help:
	@printf "Current configuration: configs/default.yml\n"
	@printf "make download  Download the raw data files\n"
	@printf "make convert   Extract selected samples and export Matrix Market files\n"
	@printf "make explore   Run the exploratory data analysis notebook\n"
	@printf "make preprocess  Run preprocessing and its notebook\n"
	@printf "make representation  Fit representations, run visualization tuning, and execute its notebook\n"

download:
	python scripts/download-data.py

convert:
	Rscript scripts/convert-raw-counts.R

explore:
	jupyter nbconvert --to notebook --execute --inplace notebooks/01_data-exploration.ipynb --ExecutePreprocessor.timeout=1200

preprocess:
	python scripts/preprocess-data.py
	jupyter nbconvert --to notebook --execute --inplace notebooks/02_quality-control-and-preprocessing.ipynb --ExecutePreprocessor.timeout=1200

representation:
	python scripts/fit-pca.py
	python scripts/train-autoencoder.py
	python scripts/tune-visualizations.py
	jupyter nbconvert --to notebook --execute --inplace notebooks/03_dimension-reduction-and-representation-learning.ipynb --ExecutePreprocessor.timeout=1200
