import os

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import yaml
from anndata import read_h5ad


CONFIG_FILE = "configs/default.yml"
INPUT_FILE = "data/processed/lung_tumor_hvg.h5ad"
CHECKPOINT_FILE = "outputs/models/autoencoder.pt"
HISTORY_FILE = "outputs/tables/autoencoder-training-history.csv"


class Autoencoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim)
        )

    def forward(self, inputs):
        return self.decoder(self.encoder(inputs))


def split_indices(samples, validation_fraction, seed):
    generator = np.random.default_rng(seed)
    training_indices = []
    validation_indices = []

    # Keep every sample represented in both training and validation data.
    for sample in pd.unique(samples):
        indices = np.flatnonzero(samples == sample)
        generator.shuffle(indices)
        n_validation = max(1, round(validation_fraction * len(indices)))
        validation_indices.extend(indices[:n_validation])
        training_indices.extend(indices[n_validation:])

    return np.array(training_indices), np.array(validation_indices)


def evaluate_loss(model, matrix, indices, batch_size, loss_function, device):
    total_loss = 0
    model.eval()

    with torch.no_grad():
        # Convert only the current sparse batch to a dense tensor for reconstruction.
        for start in range(0, len(indices), batch_size):
            batch_indices = indices[start:start + batch_size]
            batch = torch.from_numpy(matrix[batch_indices].toarray()).to(device)
            loss = loss_function(model(batch), batch)
            total_loss += loss.item() * len(batch_indices)

    return total_loss / len(indices)


def encode_cells(model, matrix, batch_size, latent_dim, device):
    embedding = np.empty((matrix.shape[0], latent_dim), dtype=np.float32)
    model.eval()

    with torch.no_grad():
        # Encode all cells in batches after the best model has been selected.
        for start in range(0, matrix.shape[0], batch_size):
            stop = min(start + batch_size, matrix.shape[0])
            batch = torch.from_numpy(matrix[start:stop].toarray()).to(device)
            embedding[start:stop] = model.encoder(batch).cpu().numpy()

    return embedding


def main():
    with open(CONFIG_FILE) as file:
        config = yaml.safe_load(file)

    settings = config["representation"]
    seed = config["analysis"]["random_seed"]
    latent_dim = settings["latent_dim"]

    os.makedirs("outputs/embeddings", exist_ok=True)
    os.makedirs("outputs/models", exist_ok=True)
    os.makedirs("outputs/tables", exist_ok=True)

    torch.manual_seed(seed)
    device = torch.device("cpu")

    # Keep the shared input sparse until each training batch is converted to a tensor.
    adata = read_h5ad(INPUT_FILE)
    matrix = adata.X.tocsr()
    samples = adata.obs["sample"].astype(str).to_numpy()
    training_indices, validation_indices = split_indices(
        samples,
        settings["validation_fraction"],
        seed
    )

    model = Autoencoder(
        matrix.shape[1],
        settings["autoencoder_hidden_dim"],
        latent_dim
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=settings["learning_rate"])
    loss_function = nn.MSELoss()
    generator = np.random.default_rng(seed)

    best_validation_loss = np.inf
    epochs_without_improvement = 0
    history = []

    for epoch in range(1, settings["max_epochs"] + 1):
        # Shuffle training cells each epoch without changing the validation split.
        shuffled_indices = generator.permutation(training_indices)
        training_loss = 0
        model.train()

        for start in range(0, len(shuffled_indices), settings["batch_size"]):
            batch_indices = shuffled_indices[start:start + settings["batch_size"]]
            batch = torch.from_numpy(matrix[batch_indices].toarray()).to(device)

            optimizer.zero_grad()
            loss = loss_function(model(batch), batch)
            loss.backward()
            optimizer.step()
            training_loss += loss.item() * len(batch_indices)

        training_loss /= len(training_indices)
        validation_loss = evaluate_loss(
            model,
            matrix,
            validation_indices,
            settings["batch_size"],
            loss_function,
            device
        )
        history.append({
            "epoch": epoch,
            "training_mse": training_loss,
            "validation_mse": validation_loss
        })

        if validation_loss < best_validation_loss:
            best_validation_loss = validation_loss
            epochs_without_improvement = 0
            torch.save(model.state_dict(), CHECKPOINT_FILE)
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= settings["patience"]:
            break

    history = pd.DataFrame(history)
    history.to_csv(HISTORY_FILE, index=False)

    # Reload the best checkpoint before exporting representations for all cells.
    model.load_state_dict(torch.load(CHECKPOINT_FILE, weights_only=True))
    embedding = encode_cells(
        model,
        matrix,
        settings["batch_size"],
        latent_dim,
        device
    )

    columns = [f"autoencoder_{number:02d}" for number in range(1, latent_dim + 1)]
    embedding = pd.DataFrame(embedding, index=adata.obs_names, columns=columns)
    embedding.index.name = "cell_id"
    output_file = f"outputs/embeddings/autoencoder-{latent_dim}.csv.gz"
    embedding.to_csv(output_file, compression="gzip")

    best_epoch = history.loc[history["validation_mse"].idxmin(), "epoch"]
    print(f"Saved {output_file}")
    print(f"Best validation epoch: {best_epoch}")


if __name__ == "__main__":
    main()
