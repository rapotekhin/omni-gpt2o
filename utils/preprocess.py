import os
import json
from PIL import Image
from torchvision import transforms
import torch
from tqdm import tqdm

def prepare_dataset(image_dir, output_dir, descriptions_file):
    """
    Prepare the dataset by:
    1. Resizing images to standard size
    2. Creating a descriptions file
    3. Organizing the data structure
    """
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, "images"), exist_ok=True)
    
    # Image preprocessing
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    # Process images
    image_files = [f for f in os.listdir(image_dir) if f.endswith(('.jpg', '.png', '.jpeg'))]
    
    print("Processing images...")
    for img_file in tqdm(image_files):
        img_path = os.path.join(image_dir, img_file)
        try:
            image = Image.open(img_path).convert('RGB')
            processed_image = transform(image)
            
            # Save processed image
            output_path = os.path.join(output_dir, "images", img_file)
            torch.save(processed_image, output_path.replace('.jpg', '.pt').replace('.png', '.pt'))
            
        except Exception as e:
            print(f"Error processing {img_file}: {str(e)}")
    
    # Create descriptions file
    if descriptions_file:
        with open(descriptions_file, 'r', encoding='utf-8') as f:
            descriptions = f.readlines()
        
        output_desc_path = os.path.join(output_dir, "descriptions.txt")
        with open(output_desc_path, 'w', encoding='utf-8') as f:
            for desc in descriptions:
                f.write(desc.strip() + '\n')
    
    print(f"Dataset preparation completed. Output directory: {output_dir}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Prepare dataset for training')
    parser.add_argument('--image_dir', type=str, required=True, help='Directory containing input images')
    parser.add_argument('--output_dir', type=str, required=True, help='Directory to save processed dataset')
    parser.add_argument('--descriptions_file', type=str, help='File containing image descriptions')
    
    args = parser.parse_args()
    
    prepare_dataset(args.image_dir, args.output_dir, args.descriptions_file) 