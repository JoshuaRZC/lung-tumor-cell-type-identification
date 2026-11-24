import os

import matplotlib.pyplot as plt
import pandas as pd
import yaml
from anndata import read_h5ad
from sklearn.decomposition import PCA


CONFIG_FILE = "configs/default.yml"
INPUT_FILE = "data/processed/lung_tumor_hvg.h5ad"
VARIANCE_FILE = "outputs/tables/pca-explained-variance.csv"
FIGURE_FILE = "outputs/figures/representation_pca-explained-variance.png"


def main():
    with open(CONFIG_FILE) as file:
        config = yaml.safe_load(file)

    settings = config["representation"]
    seed = config["analysis"]["random_seed"]

    os.makedirs("outputs/embeddings", exist_ok=True)
    os.makedirs("outputs/figures", exist_ok=True)
    os.makedirs("outputs/tables", exist_ok=True)

    # Centered PCA requires a dense copy of the log-normalized HVG matrix.
    adata = read_h5ad(INPUT_FILE)
    matrix = adata.X.toarray()
    n_components = min(settings["pca_diagnostic_components"], matrix.shape[1])

    pca = PCA(
        n_components=n_components,
        svd_solver="randomized",
        random_state=seed
    )
    embedding = pca.fit_transform(matrix)

    # Record the elbow diagnostic used to choose the shared latent dimension.
    variance = pd.DataFrame({
        "component": range(1, n_components + 1),
        "explained_variance_ratio": pca.explained_variance_ratio_,
        "cumulative_explained_variance": pca.explained_variance_ratio_.cumsum()
    })
    variance.to_csv(VARIANCE_FILE, index=False)

    # Save the elbow diagnostic before selecting the shared latent dimension.
    figure, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(
        variance["component"],
        variance["explained_variance_ratio"],
        color="steelblue"
    )
    axes[0].set_title("PCA explained variance")
    axes[0].set_xlabel("Principal component")
    axes[0].set_ylabel("Explained variance ratio")

    axes[1].plot(
        variance["component"],
        variance["cumulative_explained_variance"],
        color="steelblue"
    )
    axes[1].set_title("PCA cumulative explained variance")
    axes[1].set_xlabel("Principal component")
    axes[1].set_ylabel("Cumulative explained variance")

    figure.tight_layout()
    figure.savefig(FIGURE_FILE, dpi=300, bbox_inches="tight")

    latent_dim = settings["latent_dim"]
    if latent_dim is None:
        print(f"Saved {VARIANCE_FILE} and {FIGURE_FILE}")
        return

    # Export only the configured leading PCs for later representation comparisons.
    columns = [f"pca_{number:02d}" for number in range(1, latent_dim + 1)]
    selected_embedding = pd.DataFrame(
        embedding[:, :latent_dim],
        index=adata.obs_names,
        columns=columns
    )
    selected_embedding.index.name = "cell_id"
    output_file = f"outputs/embeddings/pca-{latent_dim}.csv.gz"
    selected_embedding.to_csv(output_file, compression="gzip")

    print(f"Saved {output_file}")


if __name__ == "__main__":
    main()
