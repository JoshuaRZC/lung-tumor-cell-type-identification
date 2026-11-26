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
ASSIGNMENT_FILE = "outputs/clusters/clustering-labels.csv.gz"
MARKER_GENES_FILE = "outputs/tables/annotation-cluster-marker-genes.csv"
SCORES_FILE = "outputs/tables/annotation-cluster-marker-scores.csv"
CLUSTER_LABELS_FILE = "outputs/tables/annotation-cluster-labels.csv"
ANNOTATION_FILE = "outputs/annotations/annotation-labels.csv.gz"


def load_expression(assignments, settings):
    counts = mmread(COUNTS_FILE).T.tocsr()
    cells = pd.read_csv(CELLS_FILE, sep="\t")
    genes = pd.read_csv(GENES_FILE, header=None, names=["gene"])
    qc_metrics = pd.read_csv(QC_METRICS_FILE, index_col="cell_id")
    gene_metrics = pd.read_csv(GENE_METRICS_FILE, index_col="gene")

    cell_mask = qc_metrics.loc[cells["cell_id"], "passed_qc"].to_numpy()
    gene_mask = gene_metrics.loc[genes["gene"], "passed_gene_filter"].to_numpy()
    adata = AnnData(
        X=counts[cell_mask][:, gene_mask],
        obs=cells.loc[cell_mask].set_index("cell_id"),
        var=pd.DataFrame(index=genes.loc[gene_mask, "gene"])
    )

    # Keep expression rows in the same stable order as the cluster assignments.
    if not adata.obs_names.equals(assignments.index):
        adata = adata[assignments.index].copy()

    sc.pp.normalize_total(adata, target_sum=settings["target_sum"])
    sc.pp.log1p(adata)
    return adata


def cluster_order(values):
    return sorted(values.astype(str).unique(), key=int)


def supporting_markers(marker_genes, panel, threshold):
    supported = marker_genes.loc[
        (marker_genes["gene"].isin(panel))
        & (marker_genes["logfoldchanges"] > 0)
        & (marker_genes["pvals_adj"] <= threshold),
        "gene"
    ]
    return ", ".join(supported.drop_duplicates())


def main():
    with open(CONFIG_FILE) as file:
        config = yaml.safe_load(file)

    settings = config["annotation"]
    with open(settings["marker_panel_file"]) as file:
        marker_config = yaml.safe_load(file)
    panel = marker_config["labels"]
    label_order = list(panel)

    assignments = pd.read_csv(ASSIGNMENT_FILE, index_col="cell_id")

    os.makedirs("outputs/tables", exist_ok=True)
    os.makedirs("outputs/annotations", exist_ok=True)
    adata = load_expression(assignments, config["preprocessing"])

    active_panel = {
        label: [gene for gene in genes if gene in adata.var_names]
        for label, genes in panel.items()
    }
    missing = {
        label: sorted(set(genes) - set(active_panel[label]))
        for label, genes in panel.items()
    }
    if any(missing.values()):
        raise ValueError(f"Marker genes absent after filtering: {missing}")

    # Score each candidate panel at the cell level before cluster aggregation.
    score_columns = []
    for number, (label, genes) in enumerate(active_panel.items()):
        column = f"marker_score_{number}"
        sc.tl.score_genes(
            adata,
            gene_list=genes,
            score_name=column,
            ctrl_size=settings["score_control_size"],
            n_bins=settings["score_n_bins"],
            random_state=config["analysis"]["random_seed"]
        )
        score_columns.append(column)

    marker_rows = []
    score_rows = []
    label_rows = []
    annotation_labels = pd.DataFrame(index=assignments.index)

    for assignment_column in assignments.columns:
        representation, method = assignment_column.rsplit("_", 1)
        combination = f"{representation}_{method}"
        clusters = assignments[assignment_column].astype(str)
        adata.obs["cluster"] = pd.Categorical(clusters)
        sc.tl.rank_genes_groups(
            adata,
            groupby="cluster",
            method="wilcoxon",
            n_genes=settings["n_ranked_genes"],
            use_raw=False
        )

        cluster_scores = adata.obs[score_columns].groupby(clusters, observed=False).mean()
        cluster_scores.columns = label_order

        for cluster in cluster_order(clusters):
            ranked = sc.get.rank_genes_groups_df(adata, group=cluster).head(
                settings["n_ranked_genes"]
            ).rename(columns={"names": "gene", "scores": "z_score"})
            ranked.insert(0, "rank", range(1, len(ranked) + 1))
            ranked.insert(0, "cluster", cluster)
            ranked.insert(0, "method", method)
            ranked.insert(0, "representation", representation)
            marker_rows.append(ranked)

            predicted_label = cluster_scores.loc[cluster].idxmax()
            predicted_score = cluster_scores.loc[cluster, predicted_label]
            supported = supporting_markers(
                ranked,
                active_panel[predicted_label],
                settings["fdr_threshold"]
            )

            label_rows.append({
                "representation": representation,
                "method": method,
                "combination": combination,
                "cluster": cluster,
                "predicted_label": predicted_label,
                "predicted_score": predicted_score,
                "supporting_markers": supported,
            })

            for label, score in cluster_scores.loc[cluster].items():
                score_rows.append({
                    "representation": representation,
                    "method": method,
                    "combination": combination,
                    "cluster": cluster,
                    "cell_type": label,
                    "marker_score": score
                })

        cluster_labels = pd.DataFrame(label_rows).query(
            "representation == @representation and method == @method"
        ).set_index("cluster")["predicted_label"]
        annotation_labels[assignment_column] = clusters.map(cluster_labels)

    pd.concat(marker_rows, ignore_index=True).to_csv(MARKER_GENES_FILE, index=False)
    pd.DataFrame(score_rows).to_csv(SCORES_FILE, index=False)
    pd.DataFrame(label_rows).to_csv(CLUSTER_LABELS_FILE, index=False)
    annotation_labels.to_csv(ANNOTATION_FILE, compression="gzip")
    print(f"Saved {MARKER_GENES_FILE}")
    print(f"Saved {SCORES_FILE}")
    print(f"Saved {CLUSTER_LABELS_FILE}")
    print(f"Saved {ANNOTATION_FILE}")


if __name__ == "__main__":
    main()
