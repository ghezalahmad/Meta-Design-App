import torch
import torch.nn as nn
import torch.optim as optim

class Generator(nn.Module):
    def __init__(self, latent_dim, output_dim):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, output_dim),
            nn.Tanh()
        )

    def forward(self, z):
        return self.model(z)

class Discriminator(nn.Module):
    def __init__(self, input_dim):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)


def train_gan(generator, discriminator, data_loader, epochs=100, latent_dim=10, lr=0.0002):
    criterion = nn.BCELoss()  # Binary Cross-Entropy Loss
    optimizer_g = optim.Adam(generator.parameters(), lr=lr)
    optimizer_d = optim.Adam(discriminator.parameters(), lr=lr)

    for epoch in range(epochs):
        for real_data in data_loader:
            # Ensure real_data is a tensor
            real_data = real_data[0] if isinstance(real_data, list) else real_data

            batch_size = real_data.size(0)

            # Train Discriminator
            optimizer_d.zero_grad()
            z = torch.randn(batch_size, latent_dim)  # Generate random noise
            fake_data = generator(z)  # Generate fake data
            real_labels = torch.ones(batch_size, 1)  # Real data labels
            fake_labels = torch.zeros(batch_size, 1)  # Fake data labels

            # Compute losses
            real_loss = criterion(discriminator(real_data), real_labels)
            fake_loss = criterion(discriminator(fake_data.detach()), fake_labels)
            loss_d = real_loss + fake_loss
            loss_d.backward()
            optimizer_d.step()

            # Train Generator
            optimizer_g.zero_grad()
            fake_labels = torch.ones(batch_size, 1)  # Flip labels for the generator
            loss_g = criterion(discriminator(fake_data), fake_labels)
            loss_g.backward()
            optimizer_g.step()

        print(f"Epoch [{epoch + 1}/{epochs}] | D Loss: {loss_d.item():.4f} | G Loss: {loss_g.item():.4f}")

    return generator

def generate_material_designs(generator, latent_dim, num_samples=10):
    generator.eval()  # Set generator to evaluation mode
    with torch.no_grad():
        z = torch.randn(num_samples, latent_dim)  # Random noise as input
        designs = generator(z).numpy()
    return designs
