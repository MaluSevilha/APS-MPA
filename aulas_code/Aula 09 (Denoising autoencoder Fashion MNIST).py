import pandas as pd
import torch
import torch.nn as nn

from torch.utils.data import TensorDataset, DataLoader
from torchvision import datasets
from plotnine import ggplot, aes, geom_tile, facet_wrap, facet_grid, coord_fixed, scale_fill_gradient, labs, theme_void, theme, element_blank, theme_set

torch.manual_seed(42)

theme_set(theme_void())

###

fashion_trn = datasets.FashionMNIST(
    root = "data",
    train = True,
    download = True
)

fashion_tst = datasets.FashionMNIST(
    root = "data",
    train = False,
    download = True
)

X_trn = fashion_trn.data.reshape(-1, 28 * 28).float() / 255
X_tst = fashion_tst.data.reshape(-1, 28 * 28).float() / 255

input_dim = X_trn.shape[1]

###

def image_df(X, title = "image"):
    rows = []

    for i in range(X.shape[0]):
        img = X[i].reshape(28, 28)

        for r in range(28):
            for c in range(28):
                rows.append({
                    "image": i + 1,
                    "x": c,
                    "y": 27 - r,
                    "value": img[r, c].item()
                })

    return pd.DataFrame(rows)

img_df = image_df(X_trn[:100])

(ggplot(img_df, aes(x = "x", y = "y", fill = "value")) +
    geom_tile() +
    scale_fill_gradient(low = "white", high = "black") +
    coord_fixed() +
    facet_wrap("~image", ncol = 10) +
    labs(title = "Fashion MNIST") +
    theme(legend_position = "none",
          strip_text = element_blank())).draw()

###

dataset = TensorDataset(X_trn, X_trn)

trn_loader = DataLoader(
    dataset,
    batch_size = 256,
    shuffle = True
)

DAE = nn.Sequential(
    nn.Linear(input_dim, 128),
    nn.ReLU(),
    nn.Linear(128, 64),
    nn.ReLU(),
    nn.Linear(64, 128),
    nn.ReLU(),
    nn.Linear(128, input_dim),
    nn.Sigmoid()
)

print(DAE)

loss_fn = nn.BCELoss()

optimizer = torch.optim.Adam(DAE.parameters(), lr = 0.001)

epochs = 20
noise_sd = 0.25

for epoch in range(epochs):
    DAE.train()
    epoch_loss = 0

    for x_batch, y_batch in trn_loader:
        x_noisy = x_batch + noise_sd * torch.randn_like(x_batch)
        x_noisy = torch.clamp(x_noisy, 0, 1)

        y_hat = DAE(x_noisy)
        loss = loss_fn(y_hat, y_batch)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item() * x_batch.shape[0]

    epoch_loss = epoch_loss / len(dataset)

    print(f"epoch: {epoch + 1:02d} | train loss: {epoch_loss:.6f}")

###

idx = torch.randperm(X_tst.shape[0])[:10]

original = X_tst[idx]

noisy = original + noise_sd * torch.randn_like(original)
noisy = torch.clamp(noisy, 0, 1)

DAE.eval()

with torch.no_grad():
    denoised = DAE(noisy)

###

def comparison_df(original, noisy, denoised):
    rows = []

    images = {
        "Original": original,
        "Noisy": noisy,
        "Denoised": denoised
    }

    for kind, X in images.items():
        for i in range(X.shape[0]):
            img = X[i].reshape(28, 28)

            for r in range(28):
                for c in range(28):
                    rows.append({
                        "kind": kind,
                        "image": i + 1,
                        "x": c,
                        "y": 27 - r,
                        "value": img[r, c].item()
                    })

    return pd.DataFrame(rows)

cmp_df = comparison_df(original, noisy, denoised)

cmp_df["kind"] = pd.Categorical(
    cmp_df["kind"],
    categories = ["Original", "Noisy", "Denoised"],
    ordered = True
)

(ggplot(cmp_df, aes(x = "x", y = "y", fill = "value")) +
    geom_tile() +
    scale_fill_gradient(low = "white", high = "black") +
    coord_fixed() +
    facet_grid("kind ~ image") +
    labs(title = "Denoising Autoencoder") +
    theme(legend_position = "none",
          axis_title = element_blank(),
          axis_text = element_blank(),
          axis_ticks = element_blank())).draw()
