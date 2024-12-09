import torch
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import torchvision.transforms as transforms
from dataset import MosaicDataset
from model import UNet
from utils import train_self_supervised, train_with_knowledge_distillation, save_comparison_images
import argparse

def train():
    # 参数配置
    BATCH_SIZE = 32
    EPOCHS = 100
    LEARNING_RATE = 0.001
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # 数据变换
    transform = transforms.Compose([
        transforms.Resize((128, 128)),  # 调整图像大小
        transforms.ToTensor(),          # 转为Tensor
    ])

    # 数据加载
    dataset = MosaicDataset(
        mosaic_dir="~/data/mosaic",
        original_dir="~/data/original",
        transform=transform  # 根据需求添加 transforms
    )

    train_size = int(0.01 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    # 数据加载器
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # 模型初始化
    unet_model = UNet().to(DEVICE)
    optimizer = optim.Adam(unet_model.parameters(), lr=LEARNING_RATE)
    criterion = torch.nn.MSELoss()

    # 自监督学习阶段
    train_self_supervised(unet_model, train_loader, criterion, optimizer, EPOCHS, DEVICE)

    # 保存模型
    torch.save(unet_model.state_dict(), "checkpoints/unet_self_supervised.pth")

    # 知识蒸馏阶段
    pretrained_model = UNet().to(DEVICE)
    pretrained_model.load_state_dict(torch.load("checkpoints/unet_self_supervised.pth"))
    encoder = unet_model.encoder
    decoder = unet_model.decoder
    pretrained_encoder = pretrained_model.encoder

    optimizer = optim.Adam(encoder.parameters(), lr=0.001)
    train_with_knowledge_distillation(encoder, pretrained_encoder, train_loader, criterion, optimizer, EPOCHS, DEVICE)

    # Save final model (encoder + decoder)
    final_model = UNet().to(DEVICE)
    final_model.encoder = encoder
    final_model.decoder = decoder
    torch.save(final_model.state_dict(), "checkpoints/unet_final.pth")

    # Save comparison images
    save_comparison_images(final_model, test_loader, DEVICE, filename='comparison.png')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a demoisacing model.")
    parser.add_argument('--device', type=str, default='cpu', help='Device to use for training (e.g., "cuda:0", "cuda:1", "cpu")')
    args = parser.parse_args()

    DEVICE = torch.device(args.device)
    train()