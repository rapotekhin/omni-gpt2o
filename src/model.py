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

        assert latent_dim == gpt2_hidden_size, "latent_dim must be equal to gpt2_hidden_size, got {} and {}".format(latent_dim, gpt2_hidden_size)

        # Linear layer to match dimensions if needed
        self.linear_projection = None
        if vit_hidden_size != gpt2_hidden_size:
            self.linear_projection = nn.Linear(vit_hidden_size, gpt2_hidden_size)

        # Patch size for ViT
        self.patch_size = self.vit.patch_embed.patch_size[0]
        self.num_patches = (image_size // self.patch_size) ** 2
        self.expected_patches = self.num_patches
        
        # Decoder input projection (latent space -> GPT hidden size)
        # self.decoder_input = nn.Linear(latent_dim, gpt2_hidden_size)
        
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
        
        # Заменяем линейный декодер патчей на свёрточный
        # Начальный размер тензора: [batch_size, gpt2_hidden_size * 2] (эмбеддинг + текстовый контекст)
        self.patch_decoder = nn.Sequential(
            # Первый линейный слой для преобразования вектора в начальную свёртку
            nn.Linear(gpt2_hidden_size * 2, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            # Преобразуем вектор в 3D тензор для свёрток
            # Размеры: [batch_size, 128, 2, 2]
            nn.Linear(512, 128 * 2 * 2),
            nn.Unflatten(1, (128, 2, 2)),
            # Свёрточная часть, увеличивающая разрешение
            # Transposed Conv: [batch_size, 128, 2, 2] -> [batch_size, 64, 4, 4]
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),
            # Transposed Conv: [batch_size, 64, 4, 4] -> [batch_size, 32, 8, 8]
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2),
            # Transposed Conv: [batch_size, 32, 8, 8] -> [batch_size, 16, 16, 16]
            nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(0.2),
            # Transposed Conv: [batch_size, 16, 16, 16] -> [batch_size, 3, 32, 32]
            nn.ConvTranspose2d(16, 3, kernel_size=3, stride=2, padding=1, output_padding=1),
        )
        
        # Новый слой self-attention для обработки декодированных патчей с зашумленными патчами
        self.patch_self_attention = nn.MultiheadAttention(
            embed_dim=3 * self.patch_size * self.patch_size,
            num_heads=4,
            dropout=0.1,
            batch_first=True
        )
        
        # Сверточные слои для улучшения качества декодированных патчей
        self.patch_refinement = nn.Sequential(
            nn.Conv2d(6, 64, kernel_size=3, padding=1),  # 6 channels (3 decoded + 3 noisy)
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2),
            nn.Conv2d(32, 3, kernel_size=3, padding=1),
        )

        # VAE components for latent space
        self.fc_mu = nn.Linear(vit_hidden_size, gpt2_hidden_size)
        self.fc_var = nn.Linear(vit_hidden_size, gpt2_hidden_size)

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
    
    def decode_image(self, z, text_embeddings, noisy_image=None):
        """
        Decode patch embeddings to image with text context and optional noisy image
        
        Args:
            z: Patch embeddings [batch_size, num_patches, latent_dim]
            text_embeddings: Text embeddings [batch_size, text_seq_len, embed_dim]
            noisy_image: Optional noisy image [batch_size, 3, H, W] to guide generation
            
        Returns:
            Reconstructed image [batch_size, 3, H, W]
        """
        batch_size = z.size(0)
        
        # Create a text context vector by averaging text embeddings
        text_context = text_embeddings.mean(dim=1, keepdim=True)  # [batch_size, 1, embed_dim]

        # Подготовим зашумленные патчи, если они предоставлены
        noisy_patches = None
        if noisy_image is not None:
            # Разбить noisy_image на патчи
            noisy_patches = []
            h = w = int(self.image_size // self.patch_size)
            
            for i in range(h):
                for j in range(w):
                    # Извлекаем патч из noisy_image
                    y_start = i * self.patch_size
                    y_end = (i + 1) * self.patch_size
                    x_start = j * self.patch_size
                    x_end = (j + 1) * self.patch_size
                    
                    patch = noisy_image[:, :, y_start:y_end, x_start:x_end]  # [batch_size, 3, patch_size, patch_size]
                    noisy_patches.append(patch)

        # Process each patch with position and text context
        decoded_patches = []
        for p in range(self.expected_patches):
            # Get position embedding for this patch
            pos_emb = self.patch_pos_embedding[:, p:p+1, :].expand(batch_size, -1, -1)
            
            # Get patch embedding for this position
            patch_emb = z[:, p:p+1, :]
            
            # Add position embedding
            patch_emb = patch_emb + pos_emb
            
            # Concatenate with text context for rich contextual information
            patch_with_context = torch.cat([patch_emb, text_context], dim=2)
            
            # Decode patch to pixels используя свёрточный декодер
            decoded_patch = self.patch_decoder(patch_with_context.view(batch_size, -1))
            
            # Reshape not needed, свёрточный декодер уже выдаёт тензор [batch_size, 3, patch_size, patch_size]
            patch_pixels = decoded_patch
            
            # Если есть зашумленный патч, добавляем информацию из него
            if noisy_patches is not None:
                # Получаем соответствующий зашумленный патч
                noisy_patch = noisy_patches[p]
                
                # Конкатенируем каналы для сверточной обработки
                combined_patch = torch.cat([patch_pixels, noisy_patch], dim=1)  # [batch_size, 6, patch_size, patch_size]
                
                # Применяем сверточную сеть для уточнения патча
                refined_patch = self.patch_refinement(combined_patch)
                
                # Сохраняем обработанный патч
                decoded_patches.append(refined_patch)
            else:
                decoded_patches.append(patch_pixels)
        
        # Если у нас есть зашумленные патчи, применяем self-attention между всеми патчами
        if noisy_patches is not None and len(decoded_patches) > 1:
            # Преобразуем список патчей в последовательность для self-attention
            # [batch_size, num_patches, 3 * patch_size * patch_size]
            patches_seq = []
            for patch in decoded_patches:
                flat_patch = patch.reshape(batch_size, 3 * self.patch_size * self.patch_size, 1)
                patches_seq.append(flat_patch)
            
            patches_seq = torch.cat(patches_seq, dim=2).transpose(1, 2)  # [batch_size, num_patches, features]
            
            # Применяем self-attention
            attended_patches, _ = self.patch_self_attention(
                query=patches_seq,
                key=patches_seq,
                value=patches_seq
            )
            
            # Преобразуем обратно в список патчей
            decoded_patches = []
            for p in range(self.expected_patches):
                patch_flat = attended_patches[:, p, :]
                patch = patch_flat.view(batch_size, 3, self.patch_size, self.patch_size)
                decoded_patches.append(patch)
        
        # Assemble patches into full image
        h = w = int(self.image_size // self.patch_size)
        rows = []
        
        for i in range(h):
            row_patches = []
            for j in range(w):
                patch_idx = i * w + j
                row_patches.append(decoded_patches[patch_idx])
            
            # Concatenate patches in the row
            row = torch.cat(row_patches, dim=3)  # Concat along width dimension
            rows.append(row)
        
        # Concatenate rows to form the final image
        image = torch.cat(rows, dim=2)  # Concat along height dimension
        
        return torch.sigmoid(image)  # Ensure pixel values are in [0, 1]
    
    def forward(self, text, original_image, noisy_image, kl_weight=0.01):
        """
        Forward pass with VAE components and token-wise processing
        
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
        combined_embeddings = torch.cat([text_embeddings, eos_token_embedding, noisy_vit_embeddings], dim=1)

        # Подготовка для хранения результатов обработки каждого токена
        generated_embeddings = []  # Для накопления сгенерированных эмбеддингов
        mu_list = []  # Для накопления mu
        log_var_list = []  # Для накопления log_var
        z_list = []  # Для накопления samplez
        decoded_patches = []  # Для накопления декодированных патчей
        
        # Разделение зашумленного изображения на патчи
        noisy_patches = None
        if noisy_image is not None:
            noisy_patches = []
            h = w = int(self.image_size // self.patch_size)
            
            for i in range(h):
                for j in range(w):
                    y_start = i * self.patch_size
                    y_end = (i + 1) * self.patch_size
                    x_start = j * self.patch_size
                    x_end = (j + 1) * self.patch_size
                    
                    patch = noisy_image[:, :, y_start:y_end, x_start:x_end]
                    noisy_patches.append(patch)
        
        # Авторегрессивно генерируем эмбеддинги и сразу их обрабатываем
        current_context = combined_embeddings
        
        for p in range(self.expected_patches):
            # 1. Генерация следующего токена
            outputs = self.gpt2(inputs_embeds=current_context)
            next_token_logits = outputs.logits[:, -1, :]  # [batch, vocab_size]
            
            # Преобразование логитов в эмбеддинг
            next_token_embedding = torch.matmul(
                torch.softmax(next_token_logits, dim=-1),
                self.gpt2.transformer.wte.weight
            ).unsqueeze(1)  # [batch, 1, hidden_size]
            
            # Добавляем в список
            generated_embeddings.append(next_token_embedding)
            
            # 2. Добавление позиционного эмбеддинга
            pos_emb = self.patch_pos_embedding[:, p:p+1, :].expand(batch_size, -1, -1)
            next_token_with_pos = next_token_embedding + pos_emb
            
            # 3. Применение cross-attention: токен слушает текст
            attended_token, _ = self.cross_attention(
                query=next_token_with_pos,
                key=text_embeddings,
                value=text_embeddings
            )
            
            # 4. Вычисление компонентов VAE
            mu_token = self.fc_mu(attended_token)  # [batch, 1, latent_dim]
            log_var_token = self.fc_var(attended_token)  # [batch, 1, latent_dim]
            
            # Сохраняем для подсчета потерь
            mu_list.append(mu_token)
            log_var_list.append(log_var_token)
            
            # 5. Репараметризация для каждого токена
            z_token = self.reparameterize(mu_token, log_var_token)  # [batch, 1, latent_dim]
            z_list.append(z_token)
            
            # 6. Декодирование патча из латентного представления
            # Создаем текстовый контекст
            text_context = text_embeddings.mean(dim=1, keepdim=True)  # [batch_size, 1, embed_dim]
            
            # Подготовка к декодированию
            patch_with_context = torch.cat([z_token, text_context], dim=2)
            
            # Декодируем патч
            decoded_patch = self.patch_decoder(patch_with_context.view(batch_size, -1))
            
            # 7. Если есть зашумленный патч, используем его для улучшения
            if noisy_patches is not None:
                noisy_patch = noisy_patches[p]  # Берем соответствующий патч
                
                # Объединяем декодированный и зашумленный патчи
                combined_patch = torch.cat([decoded_patch, noisy_patch], dim=1)  # [batch, 6, patch_size, patch_size]
                
                # Применяем сеть уточнения
                refined_patch = self.patch_refinement(combined_patch)
                
                # Сохраняем обработанный патч
                decoded_patches.append(refined_patch)
            else:
                decoded_patches.append(decoded_patch)
            
            # Обновляем контекст для следующей итерации
            current_context = torch.cat([current_context, next_token_embedding], dim=1)
        
        # Объединяем результаты всех токенов
        predicted_image_embeddings = torch.cat(generated_embeddings, dim=1)  # [batch, num_patches, embed_dim]
        mu = torch.cat(mu_list, dim=1)  # [batch, num_patches, latent_dim]
        log_var = torch.cat(log_var_list, dim=1)  # [batch, num_patches, latent_dim]
        z = torch.cat(z_list, dim=1)  # [batch, num_patches, latent_dim]
        
        # Применяем self-attention между всеми патчами если их больше одного
        if len(decoded_patches) > 1:
            # Подготавливаем последовательность патчей для self-attention
            patches_seq = []
            for patch in decoded_patches:
                flat_patch = patch.reshape(batch_size, 3 * self.patch_size * self.patch_size, 1)
                patches_seq.append(flat_patch)
            
            patches_seq = torch.cat(patches_seq, dim=2).transpose(1, 2)  # [batch, num_patches, features]
            
            # Применяем self-attention
            attended_patches, _ = self.patch_self_attention(
                query=patches_seq,
                key=patches_seq,
                value=patches_seq
            )
            
            # Преобразуем обратно в список патчей
            decoded_patches = []
            for p in range(self.expected_patches):
                patch_flat = attended_patches[:, p, :]
                patch = patch_flat.view(batch_size, 3, self.patch_size, self.patch_size)
                decoded_patches.append(patch)
        
        # Собираем полное изображение из патчей
        h = w = int(self.image_size // self.patch_size)
        rows = []
        
        for i in range(h):
            row_patches = []
            for j in range(w):
                patch_idx = i * w + j
                row_patches.append(decoded_patches[patch_idx])
            
            # Объединяем патчи в ряд
            row = torch.cat(row_patches, dim=3)  # вдоль ширины
            rows.append(row)
        
        # Объединяем ряды в изображение
        predicted_image = torch.cat(rows, dim=2)  # вдоль высоты
        predicted_image = torch.sigmoid(predicted_image)  # Гарантируем, что значения пикселей в [0, 1]
        
        # Вычисляем потери
        # 1. MSE между предсказанными и оригинальными эмбеддингами
        mse_loss = torch.nn.functional.mse_loss(
            predicted_image_embeddings,
            original_vit_embeddings[:, 1:, :]  # Исключаем class token
        )
        
        # 2. KL дивергенция для VAE
        kl_loss = self.kl_loss(mu, log_var)
        
        # 3. Потеря реконструкции
        reconstruction_loss = torch.nn.functional.mse_loss(
            predicted_image,
            original_image
        )
        
        # Комбинируем потери с весами
        total_loss = mse_loss + kl_weight * kl_loss + reconstruction_loss
        
        # Создаем словарь выходных значений
        outputs = {
            'mse_loss': mse_loss,
            'kl_loss': kl_loss, 
            'reconstruction_loss': reconstruction_loss,
            'predicted_embeddings': predicted_image_embeddings,
            'predicted_image': predicted_image,
            'mu': mu,
            'log_var': log_var,
            'sampled_z': z
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

