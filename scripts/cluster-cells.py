import os
import time

import numpy as np
import pandas as pd
import scanpy as sc
import yaml
from anndata import AnnData
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.neighbors import KNeighborsTransformer


CONFIG_FILE = "configs/default.yml"
DIAGNOSTIC_FILE = "outputs/tables/clustering-parameter-diagnostics.csv"
ASSIGNMENT_FILE = "outputs/clusters/clustering-labels.csv.gz"
SUMMARY_FILE = "outputs/tables/clustering-summary.csv"


def clustering_diagnostics(embedding, labels, sample_size, seed):
    cluster_sizes = pd.Series(labels).value_counts()
    n_clusters = len(cluster_sizes)

    if n_clusters < 2 or n_clusters == len(labels):
        silhouette = np.nan
    else:
        silhouette = silhouette_score(
            embedding,
            labels,
            sample_size=min(sample_size, len(labels)),
            random_state=seed
        )

    return {
        "n_clusters": n_clusters,
        "silhouette_score": silhouette,
        "min_cluster_size": cluster_sizes.min(),
        "median_cluster_size": cluster_sizes.median()
    }


def run_kmeans(embedding, n_clusters, sample_size, seed):
    start = time.perf_counter()
    model = KMeans(n_clusters=n_clusters, n_init=20, random_state=seed)
    labels = model.fit_predict(embedding)
    diagnostics = clustering_diagnostics(embedding, labels, sample_size, seed)
    diagnostics["runtime_seconds"] = time.perf_counter() - start
    return labels, diagnostics


def main():
    with open(CONFIG_FILE) as file:
        config = yaml.safe_load(file)

    settings = config["clustering"]
    seed = config["analysis"]["random_seed"]
    latent_dim = config["representation"]["latent_dim"]

    os.makedirs("outputs/clusters", exist_ok=True)
    os.makedirs("outputs/tables", exist_ok=True)

    embeddings = {
        "pca": pd.read_csv(
            f"outputs/embeddings/pca-{latent_dim}.csv.gz",
            index_col="cell_id"
        ),
        "autoencoder": pd.read_csv(
            f"outputs/embeddings/autoencoder-{latent_dim}.csv.gz",
            index_col="cell_id"
        )
    }
    embeddings["autoencoder"] = embeddings["autoencoder"].loc[embeddings["pca"].index]

    results = []
    assignments = {}

    for representation, frame in embeddings.items():
        embedding = frame.to_numpy()

        # Evaluate the same K-Means candidates on both learned representations.
        for n_clusters in settings["kmeans_candidates"]:
            labels, diagnostics = run_kmeans(
                embedding,
                n_clusters,
                settings["silhouette_sample_size"],
                seed
            )
            results.append({
                "representation": representation,
                "method": "kmeans",
                "parameter": n_clusters,
                **diagnostics
            })
            assignments[(representation, "kmeans", n_clusters)] = labels

        # Build a shared-neighborhood graph before comparing Leiden resolutions.
        leiden_adata = AnnData(X=embedding)
        transformer = KNeighborsTransformer(
            n_neighbors=settings["leiden_n_neighbors"],
            mode="distance",
            metric="euclidean",
            n_jobs=1
        )
        sc.pp.neighbors(
            leiden_adata,
            use_rep="X",
            transformer=transformer
        )
        for resolution in settings["leiden_resolutions"]:
            start = time.perf_counter()
            sc.tl.leiden(
                leiden_adata,
                resolution=resolution,
                random_state=seed,
                flavor="leidenalg",
                key_added="cluster"
            )
            labels = leiden_adata.obs["cluster"].astype(str).to_numpy()
            diagnostics = clustering_diagnostics(
                embedding,
                labels,
                settings["silhouette_sample_size"],
                seed
            )
            diagnostics["runtime_seconds"] = time.perf_counter() - start
            results.append({
                "representation": representation,
                "method": "leiden",
                "parameter": resolution,
                **diagnostics
            })
            assignments[(representation, "leiden", resolution)] = labels

    diagnostics = pd.DataFrame(results)
    diagnostics.to_csv(DIAGNOSTIC_FILE, index=False)
    print(f"Saved {DIAGNOSTIC_FILE}")

    final_parameters = settings["final_parameters"]

    final_keys = {
        "pca_kmeans": ("pca", "kmeans", final_parameters["pca"]["kmeans_clusters"]),
        "pca_leiden": ("pca", "leiden", final_parameters["pca"]["leiden_resolution"]),
        "autoencoder_kmeans": (
            "autoencoder",
            "kmeans",
            final_parameters["autoencoder"]["kmeans_clusters"]
        ),
        "autoencoder_leiden": (
            "autoencoder",
            "leiden",
            final_parameters["autoencoder"]["leiden_resolution"]
        )
    }
    labels = pd.DataFrame(
        {name: assignments[key] for name, key in final_keys.items()},
        index=embeddings["pca"].index
    )
    labels.index.name = "cell_id"
    labels.to_csv(ASSIGNMENT_FILE, compression="gzip")

    final_summary = pd.DataFrame([
        diagnostics.loc[
            (diagnostics["representation"] == representation)
            & (diagnostics["method"] == method)
            & (diagnostics["parameter"] == parameter)
        ].iloc[0]
        for representation, method, parameter in final_keys.values()
    ])
    final_summary.to_csv(SUMMARY_FILE, index=False)
    print(f"Saved {ASSIGNMENT_FILE}")
    print(f"Saved {SUMMARY_FILE}")


if __name__ == "__main__":
    main()
