import torch
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from dataset import MosaicDataset
import argparse
from torchvision import transforms
import dnnlib
import legacy
import os
import numpy as np
from PIL import Image

# Load pre-trained StyleGAN model (you'll need the path to the checkpoint)
def load_stylegan_model(path):
    # Load the model using StyleGAN's legacy code
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    with dnnlib.util.open_url(path) as f:
        G = legacy.load_network_pkl(f)['G_ema'].to(device)  # Use the generator part of StyleGAN
    return G

def optimize_latent(G, mosaic_img, target_img, latent_dim=512, lr=0.1, epochs=1000, device='cuda'):
    # Start with a random latent vector
    z = torch.randn([1, latent_dim], device=device, requires_grad=True)
    optimizer = optim.Adam([z], lr=lr)

    # Define MSE loss
    mse_loss = torch.nn.MSELoss()

    for epoch in range(epochs):
        optimizer.zero_grad()

        # Generate the image from the current latent vector z
        generated_img = G(z)

        # Compute the loss (MSE between generated image and target image)
        loss = mse_loss(generated_img, target_img)
        
        # Backpropagate the loss and optimize the latent vector
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 100 == 0:
            print(f'Epoch {epoch+1}/{epochs}, Loss: {loss.item()}')

    return z

def train():
    # Hyperparameters and setup
    BATCH_SIZE = 32
    EPOCHS = 100
    LEARNING_RATE = 0.001
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # Dataset transformation
    transform = transforms.Compose([
        transforms.Resize((128, 128)),  # Resize the images
        transforms.ToTensor(),          # Convert images to tensor
    ])

    # Dataset and DataLoader
    dataset = MosaicDataset(
        mosaic_dir="data/mosaic",
        original_dir="data/original",
        transform=transform
    )

    train_size = int(0.01 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # Load the pre-trained StyleGAN model
    G = load_stylegan_model('path_to_pretrained_stylegan.pkl')

    # Train loop
    for epoch in range(EPOCHS):
        for i, (mosaic, target) in enumerate(train_loader):
            # Move data to the correct device
            mosaic = mosaic.to(DEVICE)
            target = target.to(DEVICE)

            # Optimize the latent code for the mosaic image to match the target
            z = optimize_latent(G, mosaic, target, lr=LEARNING_RATE, epochs=1000, device=DEVICE)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a photo upsampling model.")
    parser.add_argument('--device', type=str, default='cuda', help='Device to use for training (e.g., "cuda:0", "cpu")')
    args = parser.parse_args()

    DEVICE = torch.device(args.device)
    train()
