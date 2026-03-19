from typing import cast

import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from commons.utils import get_device
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from torch.optim import AdamW
from tqdm import trange

DEVICE = get_device()

# read dataset
df = pd.read_csv("heart.csv")
# encode categorical columns
cat_cols = [c for c in df.columns if df[c].nunique() <= 5 and c != "target"]
onehot_df = pd.get_dummies(df, columns=cat_cols)
# split away target column
X = onehot_df.drop("target", axis="columns")
y = onehot_df["target"]
# perform a train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=0)
# rescale data and make it into tensors
sc = MinMaxScaler()
X_train = torch.tensor(sc.fit_transform(
    X_train), dtype=torch.float, device=DEVICE)
X_test = torch.tensor(sc.transform(X_test), dtype=torch.float, device=DEVICE)
y_train = torch.tensor(y_train.array, dtype=torch.long, device=DEVICE)
y_test = torch.tensor(y_test.array, dtype=torch.long, device=DEVICE)


class HeartClassifier(nn.Module):
    def __init__(self):
        super(HeartClassifier, self).__init__()
        self.fc1 = nn.Linear(30, 64)
        self.fc2 = nn.Linear(64, 128)
        self.fc3 = nn.Linear(128, 256)
        self.dropout = nn.Dropout(0.1)
        self.fc4 = nn.Linear(256, 2)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.dropout(x)

        return self.fc4(x)


def train_classifier(
        X_train_: torch.Tensor,
        y_train_: torch.Tensor,
        nr_epochs=200,
        seed=0) -> HeartClassifier:
    torch.manual_seed(seed)
    classifier = HeartClassifier().to(DEVICE).train()
    optimizer = AdamW(classifier.parameters(), lr=0.001)

    for _ in (pbar := trange(nr_epochs, desc="Epochs")):
        optimizer.zero_grad()
        output = classifier(X_train_)
        loss = F.cross_entropy(output, y_train_)
        loss.backward()
        optimizer.step()
        pbar.set_postfix(loss=loss.item())

    return classifier


def test_classifier(
        classifier: HeartClassifier,
        X_test_: torch.Tensor,
        y_test_: torch.Tensor) -> float:
    classifier.eval()
    output = classifier(X_test_)
    pred = torch.argmax(output, dim=1)
    test_acc = (y_test_ == pred).float().mean().item()

    return test_acc


class VAE(nn.Module):
    def __init__(self, input_dim: int, hidden_dim_1=16, hidden_dim_2=8, latent_dim=4):
        # Encoder
        super(VAE, self).__init__()
        self.latent_dim = latent_dim
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim_1),
            nn.BatchNorm1d(num_features=hidden_dim_1),
            nn.ReLU(),
            nn.Linear(hidden_dim_1, hidden_dim_2),
            nn.BatchNorm1d(num_features=hidden_dim_2),
            nn.ReLU()
        )
        self.fc_mu = nn.Linear(hidden_dim_2, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim_2, latent_dim)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim_2),
            nn.BatchNorm1d(num_features=hidden_dim_2),
            nn.ReLU(),
            nn.Linear(hidden_dim_2, hidden_dim_1),
            nn.BatchNorm1d(num_features=hidden_dim_1),
            nn.ReLU(),
            nn.Linear(hidden_dim_1, input_dim),
            # since we use minmax scaling
            nn.Sigmoid()
        )

    def encode(self, x):
        h = self.encoder(x)
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu, logvar):
        std = (0.5 * logvar).exp()
        eps = torch.randn_like(std)

        return mu + eps * std

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)

        return self.decode(z), mu, logvar


def train_vae(
        nr_epochs: int,
        X_train_: torch.Tensor,
        y_train_: torch.Tensor,
        seed=0) -> VAE:
    torch.manual_seed(seed)
    vae = VAE(X_train_.size(-1) + 1).to(DEVICE).train()
    # our small dataset is already on the desired device
    # and in our self-supervised setting "data" = "target"
    # so we can just manually split it without a dataloader
    real_data = torch.concat((X_train_, y_train_.view(-1, 1)), dim=1)
    batches = real_data.split(64, dim=0)
    optimizer = AdamW(vae.parameters(), lr=0.001)

    for _ in (pbar := trange(nr_epochs, desc="Epochs")):
        loss_sum = 0.

        for data in batches:
            optimizer.zero_grad()
            output, mu, logvar = vae(data)

            loss_mse = F.mse_loss(output, data, reduction="sum")
            loss_kld = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).sum()
            loss = loss_mse + loss_kld

            loss.backward()
            optimizer.step()
            loss_sum += loss.item()

        pbar.set_postfix(loss=loss_sum / len(batches))

    return vae


def sample_vae(vae: VAE, nr_samples: int, seed=0) -> torch.Tensor:
    torch.manual_seed(seed)
    vae.eval()

    with torch.no_grad():
        z = torch.randn(nr_samples, vae.latent_dim, device=DEVICE)
        synth = vae.decode(z)
        # map values in target column to 0/1
        synth[:, -1] = synth[:, -1] >= 0.5

    return synth


class BottomModel(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(BottomModel, self).__init__()
        self.fc1 = nn.Linear(input_dim, output_dim)
        self.fc2 = nn.Linear(output_dim, output_dim)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))

        return self.dropout(x)


class TopModel(nn.Module):
    def __init__(self, input_dim: int):
        super(TopModel, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 256)
        self.fc3 = nn.Linear(256, 2)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        # concatenate local model outputs before forward pass
        concat_outs = torch.cat(x, dim=1)
        x = F.relu(self.fc1(concat_outs))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))

        return self.dropout(x)


class VflClient:
    def __init__(
            self,  input_dim: int, output_dim: int,
            client_data: torch.Tensor,
            lr: float, seed: int):
        self.output_dim = output_dim
        torch.manual_seed(seed)
        self.model = BottomModel(input_dim, output_dim).to(DEVICE)
        self.client_data = client_data
        self.optimizer = AdamW(params=self.model.parameters(), lr=lr)
        self.output: torch.Tensor

    def forward_pass(self, inds: torch.Tensor) -> torch.Tensor:
        self.model.train()
        data = self.client_data[inds]
        self.optimizer.zero_grad()
        self.output = self.model(data)

        return self.output.detach().clone()

    def backward_pass(self, split_grad: torch.Tensor) -> None:
        self.output.backward(split_grad)
        self.optimizer.step()


class VflServer:
    def __init__(
            self, clients: list[VflClient],
            labels: torch.Tensor,
            lr: float, batch_size: int, seed: int):
        torch.manual_seed(seed)
        self.model = TopModel(sum(c.output_dim for c in clients)).to(DEVICE)
        self.clients = clients
        self.labels = labels
        self.optimizer = AdamW(params=self.model.parameters(), lr=lr)
        self.inds_batches = torch.arange(
            labels.size(0)).split(batch_size, dim=0)

    def run(self, nr_epochs: int) -> None:
        for _epoch in trange(nr_epochs, desc="Epochs"):
            for inds_batch in self.inds_batches:
                local_outs = [
                    c.forward_pass(inds_batch).requires_grad_()
                    for c in self.clients
                ]
                output = self.model(local_outs)
                loss = F.cross_entropy(output, self.labels[inds_batch])
                loss.backward()
                self.optimizer.step()

                with torch.no_grad():
                    for local_out, c in zip(local_outs, self.clients):
                        c.backward_pass(cast(torch.Tensor, local_out.grad))

    def test(self, X_test_chunks: tuple[torch.Tensor, ...], y_test_: torch.Tensor) -> float:
        self.model.eval()

        for c in self.clients:
            c.model.eval()

        local_outs = [
            c.model(X_test_chunk)
            for c, X_test_chunk in zip(self.clients, X_test_chunks)
        ]
        output = self.model(local_outs)
        pred = torch.argmax(output, dim=1)
        test_acc = (y_test_ == pred).float().mean().item()

        return test_acc
