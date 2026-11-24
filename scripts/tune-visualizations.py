import os

import numpy as np
import pandas as pd
import umap
import yaml
from sklearn.manifold import TSNE


CONFIG_FILE = "configs/default.yml"
COORDINATE_FILE = "outputs/embeddings/visualization-tuning-coordinates.npz"


def main():
    with open(CONFIG_FILE) as file:
        config = yaml.safe_load(file)

    settings = config["representation"]
    tuning = config["visualization_tuning"]
    seed = config["analysis"]["random_seed"]
    latent_dim = settings["latent_dim"]

    os.makedirs("outputs/embeddings", exist_ok=True)
    pca_embedding = pd.read_csv(
        f"outputs/embeddings/pca-{latent_dim}.csv.gz",
        index_col="cell_id"
    )
    autoencoder_embedding = pd.read_csv(
        f"outputs/embeddings/autoencoder-{latent_dim}.csv.gz",
        index_col="cell_id"
    ).loc[pca_embedding.index]

    # Apply each visualization setting to the same cells in both representations.
    embeddings = {
        "pca": pca_embedding.to_numpy(),
        "autoencoder": autoencoder_embedding.to_numpy()
    }
    coordinates = {}

    for representation, embedding in embeddings.items():
        # t-SNE settings vary global crowding and early separation.
        for perplexity in tuning["tsne_perplexities"]:
            for early_exaggeration in tuning["tsne_early_exaggerations"]:
                setting_id = f"{representation}_tsne_p{perplexity}_e{early_exaggeration}"
                projection = TSNE(
                    n_components=2,
                    perplexity=perplexity,
                    early_exaggeration=early_exaggeration,
                    max_iter=settings["tsne_max_iter"],
                    init="pca",
                    learning_rate="auto",
                    random_state=seed
                ).fit_transform(embedding)
                coordinates[setting_id] = projection.astype(np.float32)

        # UMAP settings vary neighborhood size and the spacing within groups.
        for n_neighbors in tuning["umap_n_neighbors"]:
            for min_dist in tuning["umap_min_dists"]:
                setting_id = f"{representation}_umap_n{n_neighbors}_d{str(min_dist).replace('.', '')}"
                projection = umap.UMAP(
                    n_neighbors=n_neighbors,
                    min_dist=min_dist,
                    random_state=seed,
                    n_jobs=1,
                    compatibility_layout=True
                ).fit_transform(embedding)
                coordinates[setting_id] = projection.astype(np.float32)

    # Keep full coordinates available for the notebook without tracking generated files.
    np.savez_compressed(
        COORDINATE_FILE,
        cell_id=pca_embedding.index.to_numpy(dtype=str),
        **coordinates
    )

    print(f"Saved {COORDINATE_FILE}")


if __name__ == "__main__":
    main()
