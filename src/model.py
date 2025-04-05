import torch
import torch.nn as nn
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import timm
from diffusers import StableDiffusionPipeline

class ImageTextGenerator(nn.Module):
    def __init__(self, gpt2_model_name="gpt2", vit_model_name="vit_base_patch16_224", device="cuda"):
        super().__init__()
        self.device = device
        
        # Initialize GPT-2
        self.gpt2 = GPT2LMHeadModel.from_pretrained(gpt2_model_name)
        self.tokenizer = GPT2Tokenizer.from_pretrained(gpt2_model_name)
        
        # Freeze GPT-2 encoder
        for param in self.gpt2.transformer.parameters():
            param.requires_grad = False
            
        # Initialize ViT
        self.vit = timm.create_model(vit_model_name, pretrained=True)
        
        # Freeze ViT
        for param in self.vit.parameters():
            param.requires_grad = False
            
        # Get dimensions
        gpt2_hidden_size = self.gpt2.config.hidden_size
        vit_hidden_size = self.vit.embed_dim
        
        # Linear layer to match dimensions
        self.linear_projection = nn.Linear(vit_hidden_size, gpt2_hidden_size)
        
        # Initialize Stable Diffusion
        self.sd_pipeline = StableDiffusionPipeline.from_pretrained(
            "runwayml/stable-diffusion-v1-5",
            torch_dtype=torch.float16
        ).to(device)
        
    def encode_text(self, text):
        # Tokenize and encode text with GPT-2
        inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            text_embeddings = self.gpt2.transformer(**inputs).last_hidden_state
            
        return text_embeddings
    
    def encode_image(self, image):
        # Encode image with ViT
        with torch.no_grad():
            image_embeddings = self.vit.forward_features(image)
            
        # Project to GPT-2 dimension
        image_embeddings = self.linear_projection(image_embeddings)
        return image_embeddings
    
    def generate_noisy_image(self, prompt, num_inference_steps=3, seed=42):
        # Generate noisy image using Stable Diffusion
        generator = torch.Generator(device=self.device).manual_seed(seed)
        
        with torch.no_grad():
            noisy_image = self.sd_pipeline(
                prompt,
                num_inference_steps=num_inference_steps,
                generator=generator
            ).images[0]
            
        return noisy_image
    
    def forward(self, text, original_image, noisy_image):
        # Encode text
        text_embeddings = self.encode_text(text)
        
        # Encode images
        original_embeddings = self.encode_image(original_image)
        noisy_embeddings = self.encode_image(noisy_image)
        
        # Combine embeddings
        combined_embeddings = torch.cat([text_embeddings, noisy_embeddings], dim=1)
        
        # Generate predictions
        outputs = self.gpt2(
            inputs_embeds=combined_embeddings,
            labels=original_embeddings
        )
        
        return outputs.loss, outputs.logits 