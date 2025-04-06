import torch
import torch.nn as nn
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import timm
from diffusers import StableDiffusionPipeline
from torchvision import transforms


class ImageTextGenerator(nn.Module):
    def __init__(self, 
                 gpt2_model_name="gpt2", 
                 vit_model_name="vit_base_patch32_224", 
                 device="cuda",
                 image_size=224,
                 latent_dim=768
        ):
        super().__init__()
        self.image_size = image_size
        self.device = device
        self.latent_dim = latent_dim
        
        # Initialize GPT-2
        self.gpt2 = GPT2LMHeadModel.from_pretrained(gpt2_model_name)
        self.tokenizer = GPT2Tokenizer.from_pretrained(gpt2_model_name)
        
        # Set pad token to eos token
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.gpt2.config.pad_token_id = self.tokenizer.eos_token_id
        
        # Freeze GPT-2 encoder but keep decoder trainable
        for param in self.gpt2.transformer.parameters():
            param.requires_grad = False

        # for name, param in self.gpt2.transformer.named_parameters():
        #     if 'h.' in name:  # Encoder layers
        #         param.requires_grad = False
        #     elif 'ln_f' in name or 'wte' in name or 'wpe' in name:  # Final layer norm and embeddings
        #         param.requires_grad = True

        self.sd_pipeline = None
            
        # Initialize ViT
        self.vit = timm.create_model(vit_model_name, pretrained=True)
        
        # Freeze ViT
        for param in self.vit.parameters():
            param.requires_grad = False
            
        # Get dimensions
        gpt2_hidden_size = self.gpt2.config.hidden_size
        vit_hidden_size = self.vit.embed_dim
        self.vit_embed_dim = vit_hidden_size
        
        # Linear layer to match dimensions if needed
        self.linear_projection = None
        if vit_hidden_size != gpt2_hidden_size:
            self.linear_projection = nn.Linear(vit_hidden_size, gpt2_hidden_size)

        # Patch size for ViT
        self.patch_size = self.vit.patch_embed.patch_size[0]
        self.num_patches = (image_size // self.patch_size) ** 2
        self.expected_patches = self.num_patches
        
        # Decoder input projection (latent space -> GPT hidden size)
        self.decoder_input = nn.Linear(latent_dim, gpt2_hidden_size)
        
        # Cross-attention layer for text-to-patch interactions
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=gpt2_hidden_size,
            num_heads=8,
            dropout=0.1,
            batch_first=True
        )

        # Positional embeddings for patches
        # Create fixed positional embeddings for patches
        position_ids = torch.arange(self.num_patches).unsqueeze(0)  # [1, num_patches]
        angles = torch.arange(gpt2_hidden_size).unsqueeze(0) / torch.pow(10000, torch.arange(0, 1, 1/gpt2_hidden_size).reshape(1, -1) * 2)
        pos_emb = torch.zeros(1, self.num_patches, gpt2_hidden_size)
        
        for i in range(self.num_patches):
            pos_emb[:, i, 0::2] = torch.sin(angles[:, 0::2] * position_ids[:, i].unsqueeze(-1))
            pos_emb[:, i, 1::2] = torch.cos(angles[:, 1::2] * position_ids[:, i].unsqueeze(-1))
            
        self.patch_pos_embedding = nn.Parameter(pos_emb, requires_grad=False)
        
        # Patch decoder - processes embeddings + text to pixels
        self.patch_decoder = nn.Sequential(
            nn.Linear(gpt2_hidden_size * 2, 512),  # Embedding dim + text context
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Linear(512, 1024),
            nn.LayerNorm(1024),
            nn.GELU(),
            nn.Linear(1024, 3 * self.patch_size * self.patch_size),  # 3 channel colors for patch
        )

        # VAE components for latent space
        self.fc_mu = nn.Linear(vit_hidden_size, latent_dim)
        self.fc_var = nn.Linear(vit_hidden_size, latent_dim)

    def init_sd_pipeline(self):
        self.sd_pipeline = StableDiffusionPipeline.from_pretrained(
            "runwayml/stable-diffusion-v1-5",
            torch_dtype=torch.float16
        ).to(self.device)
        
    def encode_text(self, text):
        # Tokenize and encode text with GPT-2
        inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            text_embeddings = self.gpt2.transformer(**inputs).last_hidden_state
            
        return text_embeddings
    
    def encode_image(self, image):
        # Encode image with ViT, сохраняя class token
        with torch.no_grad():
            features = self.vit.forward_features(image)
            
            # Project to match GPT-2 embedding dimension if needed
            if self.linear_projection is not None:
                features = self.linear_projection(features)
                
        return features
    
    def generate_noisy_image(self, prompt, num_inference_steps=3, seed=42, pil_to_tensor=True):
        # Generate noisy image using Stable Diffusion
        generator = torch.Generator(device=self.device).manual_seed(seed)

        if self.sd_pipeline is None:
            self.init_sd_pipeline()
        
        with torch.no_grad():
            noisy_image = self.sd_pipeline(
                prompt,
                num_inference_steps=num_inference_steps,
                generator=generator
            ).images[0]
        
        if pil_to_tensor:
            # Convert PIL image to tensor and normalize
            transform = transforms.Compose([
                transforms.Resize((self.image_size, self.image_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
            noisy_image = transform(noisy_image).unsqueeze(0).to(self.device)

        return noisy_image
    
    def kl_loss(self, mu, log_var):
        """Compute KL divergence loss"""
        # KL divergence loss across all patches
        kl_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp(), dim=(1, 2))
        return kl_loss.mean()
    
    def reparameterize(self, mu, log_var):
        """Reparameterization trick for VAE"""
        std = torch.exp(0.5 * log_var)  # Standard deviation
        eps = torch.randn_like(std)      # Random noise from normal distribution
        z = mu + eps * std               # Reparameterization
        return z
    
    def decode_image(self, z, text_embeddings):
        """
        Decode patch embeddings to image with text context
        
        Args:
            z: Patch embeddings [batch_size, num_patches, latent_dim]
            text_embeddings: Text embeddings [batch_size, text_seq_len, embed_dim]
            
        Returns:
            Reconstructed image [batch_size, 3, H, W]
        """
        batch_size = z.size(0)
        
        # Проецируем латентные векторы в размерность GPT2, если они имеют размер latent_dim
        if z.size(2) != self.gpt2.config.hidden_size:
            # Project from latent_dim to GPT2 hidden size
            z = self.decoder_input(z)
        
        # Create a text context vector by averaging text embeddings
        text_context = text_embeddings.mean(dim=1, keepdim=True)  # [batch_size, 1, embed_dim]

        # Process each patch with position and text context
        patches = []
        for p in range(self.expected_patches):
            # Get position embedding for this patch
            pos_emb = self.patch_pos_embedding[:, p:p+1, :].expand(batch_size, -1, -1)
            
            # Get patch embedding for this position
            patch_emb = z[:, p:p+1, :]
            
            # Add position embedding
            patch_emb = patch_emb + pos_emb
            
            # Concatenate with text context for rich contextual information
            patch_with_context = torch.cat([patch_emb, text_context], dim=2)
            # Decode patch to pixels
            decoded_patch = self.patch_decoder(patch_with_context.view(batch_size, -1))
            
            # Reshape to patch dimensions
            decoded_patch = decoded_patch.view(batch_size, 3, self.patch_size, self.patch_size)
            patches.append(decoded_patch)
        
        # Assemble patches into full image
        h = w = int(self.image_size // self.patch_size)
        rows = []
        
        for i in range(h):
            row_patches = []
            for j in range(w):
                patch_idx = i * w + j
                row_patches.append(patches[patch_idx])
            
            # Concatenate patches in the row
            row = torch.cat(row_patches, dim=3)  # Concat along width dimension
            rows.append(row)
        
        # Concatenate rows to form the final image
        image = torch.cat(rows, dim=2)  # Concat along height dimension
        
        return torch.sigmoid(image)  # Ensure pixel values are in [0, 1]
    
    def forward(self, text, original_image, noisy_image, kl_weight=0.01):
        """
        Forward pass with VAE components
        
        Args:
            text: Text descriptions
            original_image: Original images [B, 3, H, W]
            noisy_image: Noisy images from Stable Diffusion
            kl_weight: Weight for KL loss term
            
        Returns:
            total_loss: Combined loss value
            outputs: Dict containing loss components and generated content
        """
        # Encode text
        text_embeddings = self.encode_text(text)
        
        # Encode images, включая class token
        original_vit_embeddings = self.encode_image(original_image)
        noisy_vit_embeddings = self.encode_image(noisy_image).detach() 
        
        # Add positional embeddings to the noisy image embeddings
        noisy_vit_embeddings = torch.cat([
            noisy_vit_embeddings[:, :1, :],  # CLS
            noisy_vit_embeddings[:, 1:, :] + self.patch_pos_embedding
        ], dim=1)

        # Create separator token (EOS token embedding)
        batch_size = original_image.size(0)
        eos_token_id = self.tokenizer.eos_token_id
        eos_token_embedding = self.gpt2.transformer.wte(
            torch.tensor([eos_token_id] * batch_size, device=self.device)
        ).unsqueeze(1)  # Shape: [batch_size, 1, embedding_dim]
        
        # Concatenate text embeddings with EOS token and noisy image embeddings
        # Shapes: text_embeddings [batch_size, text_seq_len, hidden_size]
        #         eos_token_embedding [batch_size, 1, hidden_size]
        #         noisy_vit_embeddings [batch_size, img_seq_len+1, hidden_size] (+1 for class token)
        combined_embeddings = torch.cat([text_embeddings, eos_token_embedding, noisy_vit_embeddings], dim=1)

        # Auto-regressively generate exactly img_seq_len token embeddings
        generated_embeddings = []
        current_context = combined_embeddings
        
        for _ in range(self.expected_patches):
            # Get outputs for the current context
            outputs = self.gpt2(inputs_embeds=current_context)
            next_token_logits = outputs.logits[:, -1, :]  # Get logits for next token [batch, vocab_size]
            
            # Convert logits to embedding size using the embedding matrix
            next_token_embedding = torch.matmul(
                torch.softmax(next_token_logits, dim=-1),  # Apply softmax to get probabilities [batch, vocab_size]
                self.gpt2.transformer.wte.weight  # Token embedding matrix [vocab_size, hidden_size]
            ).unsqueeze(1)  # Add sequence dimension [batch, 1, hidden_size]
            
            # Add to generated list
            generated_embeddings.append(next_token_embedding)
            
            # Update context for next iteration by concatenating
            current_context = torch.cat([current_context, next_token_embedding], dim=1)
        
        # Stack the generated embeddings
        predicted_image_embeddings = torch.cat(generated_embeddings, dim=1)

        # Compute losses
        # 1. MSE loss between predicted embeddings and original embeddings
        mse_loss = torch.nn.functional.mse_loss(
            predicted_image_embeddings,
            original_vit_embeddings[:, 1:, :]  # Exclude class token
        )

        # Concatenate text context with image embeddings
        patches_with_pos = predicted_image_embeddings + self.patch_pos_embedding
        
        # Apply cross-attention: patches attend to text
        attended_embeddings, _ = self.cross_attention(
            query=patches_with_pos,
            key=text_embeddings,
            value=text_embeddings
        )

        # Calculate VAE components
        mu = self.fc_mu(attended_embeddings)  # [batch, num_patches, latent_dim]
        log_var = self.fc_var(attended_embeddings)  # [batch, num_patches, latent_dim]
        
        # 3. KL divergence loss for VAE - computed per patch
        kl_loss = self.kl_loss(mu, log_var)  # Already handles patch-wise computation
        
        # 4. Sample from latent space using reparameterization trick for each patch
        z = self.reparameterize(mu, log_var)  # [batch, num_patches, latent_dim]
        
        # 5. Decode sampled latent vectors patch by patch
        predicted_image = self.decode_image(z, text_embeddings)  # decode_image handles patch-wise decoding
        reconstruction_loss = torch.nn.functional.mse_loss(
            predicted_image,
            original_image
        )
        
        # Combine losses with weights
        total_loss = mse_loss + kl_weight * kl_loss + reconstruction_loss
        
        # Create outputs dictionary with all patch-wise components
        outputs = {
            'mse_loss': mse_loss,
            'kl_loss': kl_loss, 
            'reconstruction_loss': reconstruction_loss,
            'predicted_embeddings': predicted_image_embeddings,  # [batch, num_patches, embed_dim]
            'predicted_image': predicted_image,
            'mu': mu,  # [batch, num_patches, latent_dim]
            'log_var': log_var,  # [batch, num_patches, latent_dim] 
            'sampled_z': z  # [batch, num_patches, latent_dim]
        }
        return total_loss, outputs
    

if __name__ == "__main__":
    # Test the model
    model = ImageTextGenerator(device="cuda")
    model.to("cuda")
    text = "A beautiful landscape with a river and mountains"
    original_image = torch.randn(1, 3, 224, 224).to("cuda")
    
    print("Generating noisy image from prompt...")
    with torch.amp.autocast('cuda'):
        noisy_image = model.generate_noisy_image(text)

    print("Original image shape:", original_image.shape)
    print("Noisy image shape:", noisy_image.shape)

    print("Running forward pass to generate patch embeddings...")
    with torch.amp.autocast('cuda'):
        total_loss, outputs = model(text, original_image, noisy_image)
    
    print("Loss:", total_loss.item())
    print("Predicted embeddings shape (including class token):", outputs['predicted_embeddings'].shape)

    print("Reconstructed image shape:", outputs['predicted_image'].shape)

