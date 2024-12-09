import torch
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm
import matplotlib.pyplot as plt

# Self-supervised learning phase
def train_self_supervised(model, dataloader, criterion, optimizer, epochs=10, device="cpu"):
    model.train()
    scaler = GradScaler() if device != "cpu" else None

    for epoch in range(epochs):
        progress_bar = tqdm(dataloader, desc=f'Epoch [{epoch+1}/{epochs}]', unit='batch')
        for images, _ in progress_bar:
            images = images.to(device)
            
            optimizer.zero_grad()

            if scaler:
                with autocast():
                    outputs = model(images)
                    loss = criterion(outputs, images)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                outputs = model(images)
                loss = criterion(outputs, images)
                loss.backward()
                optimizer.step()

            progress_bar.set_postfix(loss=loss.item())
        print(f'Epoch [{epoch+1}/{epochs}] completed, Loss: {loss.item():.4f}')
    
    torch.save(model.state_dict(), "checkpoints/unet_self_supervised.pth")
    print("Self-supervised model saved.")

# Knowledge distillation phase
def train_with_knowledge_distillation(encoder, pretrained_encoder, dataloader, criterion, optimizer, epochs=10, device="cpu"):
    encoder.train()
    pretrained_encoder.eval()
    scaler = GradScaler() if device != "cpu" else None

    for epoch in range(epochs):
        progress_bar = tqdm(dataloader, desc=f'Epoch [{epoch+1}/{epochs}]', unit='batch')
        for mosaic_images, original_images in progress_bar:
            mosaic_images = mosaic_images.to(device)
            original_images = original_images.to(device)

            optimizer.zero_grad()

            with torch.no_grad():
                target_latent = pretrained_encoder(original_images)
            
            if scaler:
                with autocast():
                    encoded_mosaic = encoder(mosaic_images)
                    loss = criterion(encoded_mosaic, target_latent)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                encoded_mosaic = encoder(mosaic_images)
                loss = criterion(encoded_mosaic, target_latent)
                loss.backward()
                optimizer.step()

            progress_bar.set_postfix(loss=loss.item())
        print(f'Epoch [{epoch+1}/{epochs}] completed, Loss: {loss.item():.4f}')

def save_comparison_images(model, test_loader, device, filename='comparison.png', num_images=10):
    model.eval()
    images_so_far = 0
    fig, axes = plt.subplots(num_images, 3, figsize=(10, num_images * 3))
    with torch.no_grad():
        for i, (mosaic_images, original_images) in enumerate(test_loader):
            mosaic_images = mosaic_images.to(device)
            original_images = original_images.to(device)
            outputs = model(mosaic_images)
            
            for j in range(mosaic_images.size(0)):
                if images_so_far >= num_images:
                    break
                # Convert tensors to numpy arrays
                mosaic_image = mosaic_images[j].cpu().numpy().transpose((1, 2, 0))
                original_image = original_images[j].cpu().numpy().transpose((1, 2, 0))
                output_image = outputs[j].cpu().numpy().transpose((1, 2, 0))
                
                # Normalize images to [0, 1] range for displaying
                mosaic_image = (mosaic_image - mosaic_image.min()) / (mosaic_image.max() - mosaic_image.min())
                original_image = (original_image - original_image.min()) / (original_image.max() - original_image.min())
                output_image = (output_image - output_image.min()) / (output_image.max() - output_image.min())

                # Plot images
                axes[images_so_far, 0].imshow(mosaic_image)
                axes[images_so_far, 0].set_title('Mosaic')
                axes[images_so_far, 1].imshow(original_image)
                axes[images_so_far, 1].set_title('Original')
                axes[images_so_far, 2].imshow(output_image)
                axes[images_so_far, 2].set_title('Reconstructed')
                
                axes[images_so_far, 0].axis('off')
                axes[images_so_far, 1].axis('off')
                axes[images_so_far, 2].axis('off')
                
                images_so_far += 1
            
            if images_so_far >= num_images:
                break

    plt.tight_layout()
    plt.savefig(filename)
    plt.close(fig)
