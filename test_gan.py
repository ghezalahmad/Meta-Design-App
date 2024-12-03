from gan_model import Generator, Discriminator, train_gan, generate_material_designs
import torch
import numpy as np
from torch.utils.data import DataLoader, TensorDataset

# Dummy dataset
real_data = np.random.rand(1000, 5)  # Replace with material property data
dataset = TensorDataset(torch.tensor(real_data, dtype=torch.float32))
data_loader = DataLoader(dataset, batch_size=32, shuffle=True)

# Initialize models
latent_dim = 10
generator = Generator(latent_dim=latent_dim, output_dim=5)
discriminator = Discriminator(input_dim=5)

# Train GAN
trained_generator = train_gan(generator, discriminator, data_loader, epochs=10, latent_dim=latent_dim, lr=0.0002)

# Generate new designs
new_designs = generate_material_designs(trained_generator, latent_dim=latent_dim, num_samples=10)
print("Generated Material Designs:")
print(new_designs)
