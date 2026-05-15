import pandas as pd
import torch
import torch.nn as nn

from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import roc_curve, auc
from plotnine import ggplot, aes, geom_line, geom_abline, labs, theme_bw, theme_set, annotate

torch.manual_seed(42)

db = pd.read_csv("creditcard.csv")

print(round(100 * db["Class"].value_counts().sort_index() / db.shape[0], 2))

x = torch.tensor(db.drop(columns = ["Class"]).values, dtype = torch.float32)
y = torch.tensor(db["Class"].values, dtype = torch.int64)

x_mean = x.mean(dim = 0, keepdim = True)

x_sd = x.std(dim = 0, unbiased = True, keepdim = True)

x = (x - x_mean) / x_sd

n = x.shape[0]

idx = torch.randperm(n)

trn = idx[:round(0.8 * n)]

tst_mask = torch.ones(n, dtype = torch.bool)
tst_mask[trn] = False

x_trn = x[trn]
y_trn = y[trn]

legit = (y_trn == 0)

x_trn = x_trn[legit]
y_trn = y_trn[legit]

x_tst = x[tst_mask]
y_tst = y[tst_mask]

input_dim = x.shape[1]

dataset = TensorDataset(x_trn, x_trn)

trn_loader = DataLoader(
    dataset,
    batch_size = 256,
    shuffle = True
)

autoencoder = nn.Sequential(
    nn.Linear(input_dim, 16),
    nn.Tanh(),
    nn.Linear(16, 8),
    nn.Tanh(),
    nn.Linear(8, 16),
    nn.Tanh(),
    nn.Linear(16, input_dim)
)

print(autoencoder)

loss_fn = nn.MSELoss()

optimizer = torch.optim.Adam(autoencoder.parameters(), lr = 0.001)

epochs = 20

for epoch in range(epochs):
    autoencoder.train()
    epoch_loss = 0

    for x_batch, y_batch in trn_loader:
        y_hat = autoencoder(x_batch)
        loss = loss_fn(y_hat, y_batch)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item() * x_batch.shape[0]

    epoch_loss = epoch_loss / len(dataset)

    print(f"epoch: {epoch + 1:02d} | train loss: {epoch_loss:.6f}")

autoencoder.eval()

with torch.no_grad():
    x_hat = autoencoder(x_tst)

error = ((x_tst - x_hat)**2).mean(dim = 1)

prob = error / (1 + error) # tratando o erro como um "odds" sintético

fpr, tpr, thresholds = roc_curve(y_tst.tolist(), prob.tolist())

auc_value = auc(fpr, tpr)

theme_set(theme_bw())

(ggplot(mapping = aes(x = fpr, y = tpr)) +
    geom_line(size = 1, color = "blue") +
    geom_abline(intercept = 0, slope = 1, linetype = "dashed") +
    annotate("text", x = 0.63, y = 0.3, label = f"AUC = {auc_value:.4f}") +
    labs(x = "FPR (1 - Specificity)", y = "TPR (Sensitivity)", title = "ROC"))
