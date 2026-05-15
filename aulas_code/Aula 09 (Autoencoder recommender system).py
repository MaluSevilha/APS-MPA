import pandas as pd
import torch
import torch.nn as nn

from torch.utils.data import TensorDataset, DataLoader

torch.manual_seed(42)

db = pd.read_csv("MovieLense.csv")

user_id = db["user_id"]

X = torch.tensor(db.drop(columns = ["user_id"]).values, dtype = torch.float32)

movie_names = db.drop(columns = ["user_id"]).columns

num_users = X.shape[0]
num_items = X.shape[1]

latent_dim = 50

dataset = TensorDataset(X, X)

trn_loader = DataLoader(
    dataset,
    batch_size = 32,
    shuffle = True
)

autoencoder = nn.Sequential(
    nn.Dropout(0.2),
    nn.Linear(num_items, latent_dim),
    nn.ReLU(),
    nn.Dropout(0.2),
    nn.Linear(latent_dim, num_items)
)

print(autoencoder)

def masked_mse(y_hat, y):
    mask = (y != 0).float()
    num = (mask * (y - y_hat)**2).sum()
    den = mask.sum()
    return num / den

optimizer = torch.optim.Adam(autoencoder.parameters(), lr = 0.001)

epochs = 50

for epoch in range(epochs):
    autoencoder.train()
    epoch_loss = 0

    for x_batch, y_batch in trn_loader:
        y_hat = autoencoder(x_batch)
        loss = masked_mse(y_hat, y_batch)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item() * x_batch.shape[0]

    epoch_loss = epoch_loss / len(dataset)

    print(f"epoch: {epoch + 1:02d} | train loss: {epoch_loss:.6f}")

autoencoder.eval()

with torch.no_grad():
    X_hat = autoencoder(X)

def top_n_recom(user_idx, n = 10):
    row_idx = user_idx - 1

    rated = X[row_idx] > 0
    preds = X_hat[row_idx]

    preds = preds.clone()
    preds[rated] = -torch.inf

    top_k = torch.topk(preds, k = n).indices

    return movie_names[top_k.tolist()].tolist()

for movie in top_n_recom(user_idx = 1, n = 20):
    print(movie)
