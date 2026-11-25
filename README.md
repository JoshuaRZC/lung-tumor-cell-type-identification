# Identifying Cell Types in Lung Tumors: A Machine Learning Approach

This project develops a machine learning pipeline for identifying cell populations in lung tumors from single-cell RNA sequencing data.

The current dataset plan is GEO GSE131907, limited to 11 primary lung tumor samples.

## Project structure

The repository separates configuration, data, analysis code, outputs, and
project documents to support a reproducible workflow.

```text
lung-tumor-cell-type-identification/
├── .gitignore              # Files excluded from version control
├── README.md                # Project overview and setup instructions
├── Makefile                 # Workflow automation
├── environment.yml          # Conda environment specification
├── configs/                 # Shared experiment settings
│   └── default.yml          # Dataset selection and analysis parameters
├── data/
│   ├── raw/                 # Unmodified downloaded data
│   └── processed/           # Generated inputs for analysis
├── scripts/                 # Reproducible pipeline scripts
├── notebooks/               # Curated exploratory and presentation notebooks
├── outputs/
│   ├── figures/             # Generated figures
│   ├── tables/              # Generated tables and evaluation results
│   ├── clusters/            # Per-cell clustering assignments
│   ├── embeddings/          # Learned cell representations
│   └── models/              # Trained model artifacts
├── documents/
│   ├── proposal/            # Project proposal
│   └── report/              # Final report source and rendered files
├── references/              # Local background materials and project notes
└── instructions/            # Local course assignment requirements
```

## Environment and workflow

Create the Conda environment with:

```bash
conda env create -f environment.yml
conda activate lung-tumor-cell-type-identification
```

Use `make help` to view available workflow commands.
