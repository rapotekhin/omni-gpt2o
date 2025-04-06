import torch
from PIL import Image
import argparse
import os
from src.model import ImageTextGenerator
import matplotlib.pyplot as plt
from torchvision import transforms
import numpy as np

def generate_image(model, prompt, output_dir="generated_samples", device="cuda"):
    """
    Generate an image from a text prompt using the trained Omni-GPT2o model
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Set model to evaluation mode
    model.eval()
    
    # Generate noisy image using Stable Diffusion 1.5
    print(f"Generating noisy image for prompt: '{prompt}'")
    noisy_image = model.generate_noisy_image(prompt, num_inference_steps=3)
    
    # Calculate outputs using forward pass
    print("Generating final image...")
    with torch.no_grad():
        with torch.autocast(device_type=device, dtype=torch.bfloat16):
            # Create a dummy "original image" for the forward pass
            # We'll use the noisy image as placeholder, but it won't affect the generation
            dummy_original = noisy_image.clone()
            
            # Forward pass
            _, outputs = model([prompt], dummy_original, noisy_image)
        
        # Get the generated image
        generated_image = outputs['predicted_image']
        
        # Denormalize images for visualization
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1).to(device)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1).to(device)
        
        # Denormalize and convert to PIL
        # Noisy image
        noisy_img = noisy_image[0].cpu().detach() * std.cpu() + mean.cpu()
        noisy_img = noisy_img.permute(1, 2, 0).float().numpy()
        noisy_img = np.clip(noisy_img, 0, 1)
        noisy_pil = Image.fromarray((noisy_img * 255).astype(np.uint8))
        
        # Generated image
        gen_img = generated_image[0].cpu().detach() * std.cpu() + mean.cpu()
        gen_img = gen_img.permute(1, 2, 0).float().numpy()
        gen_img = np.clip(gen_img, 0, 1)
        gen_pil = Image.fromarray((gen_img * 255).astype(np.uint8))
    
    # Save both noisy and final images
    noisy_path = os.path.join(output_dir, f"noisy_{prompt.replace(' ', '_')[:30]}.png")
    final_path = os.path.join(output_dir, f"final_{prompt.replace(' ', '_')[:30]}.png")
    
    noisy_pil.save(noisy_path)
    gen_pil.save(final_path)
    
    print(f"Noisy image saved to {noisy_path}")
    print(f"Final generated image saved to {final_path}")
    return noisy_path, final_path

def main():
    parser = argparse.ArgumentParser(description="Generate images from text prompts with Omni-GPT2o")
    parser.add_argument("--prompt", type=str, default="A beautiful landscape with mountains and a lake",
                        help="Text prompt for image generation")
    parser.add_argument("--output_dir", type=str, default="generated_samples",
                        help="Directory to save generated images")
    parser.add_argument("--model", type=str, default="models/omni-gpt2o_latent256_best.pt",
                        help="Path to the trained model checkpoint")
    parser.add_argument("--latent_dim", type=int, default=256,
                        help="Latent dimension of the model. Use 256 for model trained with default params.")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")
    args = parser.parse_args()
    
    # Set seed for reproducibility
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Initialize model
    print(f"Initializing Omni-GPT2o model with latent dimension {args.latent_dim}...")
    model = ImageTextGenerator(device=device, latent_dim=args.latent_dim)
    model = model.to(device)
    
    # Load model checkpoint
    if os.path.exists(args.model):
        checkpoint = torch.load(args.model, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded model from {args.model}")
        print(f"Checkpoint epoch: {checkpoint.get('epoch', 'unknown')}, Loss: {checkpoint.get('loss', 'unknown')}")
    else:
        print(f"Warning: Model not found at {args.model}")
        print("Using untrained model - results will be poor")
    
    # Generate images
    noisy_path, final_path = generate_image(model, args.prompt, args.output_dir, device)
    
    # Display the images
    print("Displaying generated images...")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    
    # Load and display noisy image
    noisy_img = Image.open(noisy_path)
    ax1.imshow(np.array(noisy_img))
    ax1.set_title("Noisy Image (Stable Diffusion)")
    ax1.axis('off')
    
    # Load and display final image
    final_img = Image.open(final_path)
    ax2.imshow(np.array(final_img))
    ax2.set_title(f"Final Image (Omni-GPT2o)")
    ax2.axis('off')
    
    plt.suptitle(f"Generated from: '{args.prompt}'")
    plt.tight_layout()
    
    comparison_path = os.path.join(args.output_dir, f"comparison_{args.prompt.replace(' ', '_')[:30]}.png")
    plt.savefig(comparison_path)
    print(f"Comparison image saved to {comparison_path}")
    
    print("Done!")

if __name__ == "__main__":
    main() 