import torch
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import torchvision.transforms as transforms
from dataset import MosaicDataset
from model import UNet
from utils import train_self_supervised, train_with_knowledge_distillation, save_comparison_images
import argparse
from torch.cuda.amp import GradScaler, autocast

def train():
    # Configuration parameters
    BATCH_SIZE = 512
    EPOCHS = 10
    LEARNING_RATE = 0.01
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # Data transformations
    transform = transforms.Compose([
        transforms.Resize((128, 128)),          # Resize to 128x128
        transforms.RandomHorizontalFlip(),      # Random horizontal flip
        transforms.ToTensor(),                  # Convert to tensor and scale pixel values to [0, 1]
        transforms.Normalize((0.5,), (0.5,))    # Normalize to [-1, 1]
    ])

    # Data loading
    dataset = MosaicDataset(
        mosaic_dir="~/data/mosaic",
        original_dir="~/data/original",
        transform=transform  # Add transforms as needed
    )

    train_size = int(0.01 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    # Data loaders
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

    # Model initialization
    unet_model = UNet().to(DEVICE)

    # Use DataParallel for multi-GPU training
    if torch.cuda.device_count() > 1:
        unet_model = torch.nn.DataParallel(unet_model)

    optimizer = optim.Adam(unet_model.parameters(), lr=LEARNING_RATE)
    criterion = torch.nn.MSELoss()

    # Self-supervised learning phase
    train_self_supervised(unet_model, train_loader, criterion, optimizer, 10, DEVICE)

    # Knowledge distillation phase
    pretrained_model = UNet().to(DEVICE)
    if torch.cuda.device_count() > 1:
        pretrained_model = torch.nn.DataParallel(pretrained_model)
    pretrained_model.load_state_dict(torch.load("checkpoints/unet_self_supervised.pth"))
    encoder = unet_model.module.encoder if torch.cuda.device_count() > 1 else unet_model.encoder
    decoder = unet_model.module.decoder if torch.cuda.device_count() > 1 else unet_model.decoder
    pretrained_encoder = pretrained_model.module.encoder if torch.cuda.device_count() > 1 else pretrained_model.encoder

    optimizer = optim.Adam(encoder.parameters(), lr=0.001)
    train_with_knowledge_distillation(encoder, pretrained_encoder, train_loader, criterion, optimizer, EPOCHS, DEVICE)

    # Save final model (encoder + decoder)
    final_model = UNet().to(DEVICE)
    final_model.encoder = encoder
    final_model.decoder = decoder

    if torch.cuda.device_count() > 1:
        final_model = torch.nn.DataParallel(final_model)

    torch.save(final_model.state_dict(), "checkpoints/unet_final.pth")

    # Save comparison images
    save_comparison_images(final_model, train_loader, DEVICE, filename='comparison.png')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a demosaicing model.")
    parser.add_argument('--device', type=str, default='cpu', help='Device to use for training (e.g., "cuda:0", "cuda:1", "cpu")')
    args = parser.parse_args()

    DEVICE = torch.device(args.device)
    train()
