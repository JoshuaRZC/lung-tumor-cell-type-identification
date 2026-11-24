import os

import pandas as pd
import scanpy as sc
import yaml
from anndata import AnnData
from scipy.io import mmread


CONFIG_FILE = "configs/default.yml"
COUNTS_FILE = "data/processed/lung_tumor_raw_counts.mtx"
CELLS_FILE = "data/processed/lung_tumor_cells.tsv"
GENES_FILE = "data/processed/lung_tumor_genes.tsv"
QC_METRICS_FILE = "data/processed/lung_tumor_qc_metrics.csv.gz"
GENE_METRICS_FILE = "data/processed/lung_tumor_gene_qc_metrics.csv.gz"
HVG_METRICS_FILE = "data/processed/lung_tumor_hvg_metrics.csv.gz"
HVG_DATA_FILE = "data/processed/lung_tumor_hvg.h5ad"
SAMPLE_SUMMARY_FILE = "outputs/tables/preprocessing-sample-summary.csv"
FILTERING_SUMMARY_FILE = "outputs/tables/preprocessing-filtering-summary.csv"


def main():
    with open(CONFIG_FILE) as file:
        config = yaml.safe_load(file)

    sample_order = config["dataset"]["selected_samples"]
    settings = config["preprocessing"]

    # Load the raw Matrix Market file as a sparse cells by genes matrix.
    counts = mmread(COUNTS_FILE).T.tocsr()
    cells = pd.read_csv(CELLS_FILE, sep="\t")
    genes = pd.read_csv(GENES_FILE, header=None, names=["gene"])

    adata = AnnData(
        X=counts,
        obs=cells.set_index("cell_id"),
        var=pd.DataFrame(index=genes["gene"])
    )
    adata.obs["sample"] = pd.Categorical(
        adata.obs["sample"],
        categories=sample_order,
        ordered=True
    )
    adata.var["mt"] = adata.var_names.str.startswith("MT-")

    # Calculate cell-level UMI, detected-gene, and mitochondrial metrics.
    sc.pp.calculate_qc_metrics(
        adata,
        qc_vars=["mt"],
        percent_top=None,
        log1p=False,
        inplace=True
    )

    adata.obs["passed_qc"] = (
        (adata.obs["total_counts"] >= settings["min_total_counts"])
        & (adata.obs["total_counts"] <= settings["max_total_counts"])
        & (adata.obs["n_genes_by_counts"] >= settings["min_genes_per_cell"])
        & (adata.obs["n_genes_by_counts"] <= settings["max_genes_per_cell"])
        & (adata.obs["pct_counts_mt"] <= settings["max_mito_percent"])
    )

    qc_metrics = adata.obs[
        ["sample", "total_counts", "n_genes_by_counts", "pct_counts_mt", "passed_qc"]
    ].copy()
    qc_metrics.index.name = "cell_id"

    before_summary = (
        qc_metrics.groupby("sample", observed=False)
        .agg(
            cells_before=("sample", "size"),
            median_total_counts_before=("total_counts", "median"),
            median_detected_genes_before=("n_genes_by_counts", "median"),
            median_mito_percent_before=("pct_counts_mt", "median")
        )
    )
    after_summary = (
        qc_metrics.loc[qc_metrics["passed_qc"]]
        .groupby("sample", observed=False)
        .agg(
            cells_after=("sample", "size"),
            median_total_counts_after=("total_counts", "median"),
            median_detected_genes_after=("n_genes_by_counts", "median"),
            median_mito_percent_after=("pct_counts_mt", "median")
        )
    )
    qc_summary = before_summary.join(after_summary).reindex(sample_order)
    qc_summary["retention_percent"] = (
        100 * qc_summary["cells_after"] / qc_summary["cells_before"]
    )

    # Save cell-level metrics and the sample summary before subsetting.
    os.makedirs("outputs/tables", exist_ok=True)
    qc_metrics.to_csv(QC_METRICS_FILE, compression="gzip")
    qc_summary.round(2).to_csv(SAMPLE_SUMMARY_FILE)

    # Filter cells first, then remove genes detected in too few retained cells.
    adata = adata[adata.obs["passed_qc"]].copy()
    gene_metrics = pd.DataFrame({
        "n_cells_by_counts": adata.X.getnnz(axis=0)
    }, index=adata.var_names)
    gene_metrics.index.name = "gene"
    gene_metrics["passed_gene_filter"] = (
        gene_metrics["n_cells_by_counts"] >= settings["min_cells_per_gene"]
    )
    gene_metrics.to_csv(GENE_METRICS_FILE, compression="gzip")

    sc.pp.filter_genes(adata, min_cells=settings["min_cells_per_gene"])

    # Normalize each retained cell before selecting sample-aware HVGs.
    sc.pp.normalize_total(adata, target_sum=settings["target_sum"])
    sc.pp.log1p(adata)
    sc.pp.highly_variable_genes(
        adata,
        n_top_genes=settings["n_hvgs"],
        batch_key=settings["hvg_batch_key"],
        flavor=settings["hvg_flavor"]
    )

    hvg_metrics = adata.var.copy()
    hvg_metrics.index.name = "gene"
    hvg_metrics.to_csv(HVG_METRICS_FILE, compression="gzip")

    adata = adata[:, adata.var["highly_variable"]].copy()
    adata.write(HVG_DATA_FILE)

    filtering_summary = pd.DataFrame({
        "cells_before_filtering": [counts.shape[0]],
        "cells_after_cell_filtering": [qc_metrics["passed_qc"].sum()],
        "cell_retention_percent": [100 * qc_metrics["passed_qc"].mean()],
        "genes_before_filtering": [counts.shape[1]],
        "genes_after_gene_filtering": [gene_metrics["passed_gene_filter"].sum()],
        "selected_hvgs": [adata.n_vars]
    })
    filtering_summary.round(2).to_csv(FILTERING_SUMMARY_FILE, index=False)

    print(f"Saved {HVG_DATA_FILE}")


if __name__ == "__main__":
    main()
