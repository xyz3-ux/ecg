import torch
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn

latent_dim = 100

class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 625),
            nn.Tanh()
        )

    def forward(self, z):
        return self.model(z)

class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(625, 512),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)

if __name__ == "__main__":

    X = np.load("ecg_dataset_full.npy")  # (N, 1, 625)
    X = X.reshape(-1, 625)          # flatten for dense model
    print("Dataset shape:", X.shape)

    X_tensor = torch.tensor(X, dtype=torch.float32)

    dataset = TensorDataset(X_tensor)
    dataloader = DataLoader(dataset, batch_size=2000, shuffle=True)
    # training setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    G = Generator().to(device)
    D = Discriminator().to(device)

    criterion = nn.BCELoss()

    optimizer_G = torch.optim.Adam(G.parameters(), lr=0.0002)
    optimizer_D = torch.optim.Adam(D.parameters(), lr=0.0002)

    # training loop
    epochs = 1

    for epoch in range(epochs):
        for real_batch, in dataloader:
            real_batch = real_batch.to(device)
            batch_size = real_batch.size(0)

            # Real and fake labels
            real_labels = torch.ones(batch_size, 1).to(device)
            fake_labels = torch.zeros(batch_size, 1).to(device)

            # ---- Train Discriminator ----
            optimizer_D.zero_grad()

            # Real loss
            outputs_real = D(real_batch)
            loss_real = criterion(outputs_real, real_labels)

            # Fake loss
            z = torch.randn(batch_size, latent_dim).to(device)
            fake_data = G(z)
            outputs_fake = D(fake_data.detach())
            loss_fake = criterion(outputs_fake, fake_labels)

            loss_D = loss_real + loss_fake
            loss_D.backward()
            optimizer_D.step()

            # ---- Train Generator ----
            optimizer_G.zero_grad()

            z = torch.randn(batch_size, latent_dim).to(device)
            fake_data = G(z)
            outputs = D(fake_data)
            loss_G = criterion(outputs, real_labels)

            loss_G.backward()
            optimizer_G.step()
            print('done')

        print(f"Epoch [{epoch+1}/{epochs}]  D Loss: {loss_D.item():.4f}  G Loss: {loss_G.item():.4f}")

    G.eval()

    with torch.no_grad():
        z = torch.randn(5, latent_dim).to(device)
        fake_samples = G(z).cpu().numpy()

    print(fake_samples.shape)

    # plt.plot(fake_samples[0])
    # plt.title("Generated ECG")
    # plt.show()