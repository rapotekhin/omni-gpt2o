import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os
from tqdm import tqdm
from model import ImageTextGenerator

class ImageTextDataset(Dataset):
    def __init__(self, image_dir, text_file, transform=None):
        self.image_dir = image_dir
        self.transform = transform or transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                              std=[0.229, 0.224, 0.225])
        ])
        
        # Load text descriptions
        self.texts = []
        with open(text_file, 'r', encoding='utf-8') as f:
            for line in f:
                self.texts.append(line.strip())
                
        self.image_files = [f for f in os.listdir(image_dir) if f.endswith(('.jpg', '.png', '.jpeg'))]
        
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.image_files[idx])
        image = Image.open(img_path).convert('RGB')
        image = self.transform(image)
        
        text = self.texts[idx % len(self.texts)]  # Cycle through texts if fewer than images
        return image, text

def train(model, train_loader, optimizer, device, num_epochs=10):
    model.train()
    
    for epoch in range(num_epochs):
        total_loss = 0
        progress_bar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}')
        
        for batch_idx, (images, texts) in enumerate(progress_bar):
            images = images.to(device)
            
            # Generate noisy images
            noisy_images = []
            for text in texts:
                noisy_img = model.generate_noisy_image(text)
                noisy_images.append(noisy_img)
            noisy_images = torch.stack(noisy_images).to(device)
            
            # Forward pass
            loss, _ = model(texts, images, noisy_images)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            progress_bar.set_postfix({'loss': total_loss / (batch_idx + 1)})
            
        avg_loss = total_loss / len(train_loader)
        print(f'Epoch {epoch+1}/{num_epochs}, Average Loss: {avg_loss:.4f}')

def main():
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Initialize model
    model = ImageTextGenerator(device=device)
    model = model.to(device)
    
    # Create dataset and dataloader
    dataset = ImageTextDataset(
        image_dir="data/images",
        text_file="data/descriptions.txt"
    )
    train_loader = DataLoader(dataset, batch_size=8, shuffle=True, num_workers=4)
    
    # Initialize optimizer (only for trainable parameters)
    trainable_params = list(model.linear_projection.parameters()) + list(model.gpt2.lm_head.parameters())
    optimizer = torch.optim.AdamW(trainable_params, lr=1e-4)
    
    # Train the model
    train(model, train_loader, optimizer, device)

if __name__ == "__main__":
    main() 