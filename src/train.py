import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import pandas as pd
import os
import argparse
import time
import random
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.tensorboard import SummaryWriter

# Функция для установки всех сидов
def set_seed(seed=42):
    """
    Устанавливает seed для всех генераторов случайных чисел для воспроизводимости результатов.
    
    Args:
        seed: значение seed, по умолчанию 42
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"Seed установлен на {seed} для воспроизводимости результатов")

# Импортируем ClearML
try:
    from clearml import Task
except ImportError:
    Task = None

import sys
sys.path.append('.')

from src.model import ImageTextGenerator
from utils.noise import diffuse_silhouette_effect_strong

class ImageTextDataset(Dataset):
    def __init__(self, image_dir, text_file, transform=None, max_samples=None):
        """
        Parameters
        ----------
        image_dir : _type_
            000000179765.jpg
            000000179765.jpg
            000000190236.jpg
        text_file : _type_
            000000179765.jpg|A black Honda motorcycle parked in front of a garage.
            000000179765.jpg|A Honda motorcycle parked in a grass driveway
            000000190236.jpg|An office cubicle with four different types of computers.
        """
        self.image_dir = image_dir
        self.transform = transform or transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                              std=[0.229, 0.224, 0.225])
        ])
        
        # Load text descriptions
        self.texts = pd.read_csv(text_file, header=None, sep='|')
        self.texts.columns = ['image_name', 'text']

        self.image_files = [f for f in os.listdir(image_dir) if f.endswith(('.jpg', '.png', '.jpeg'))]
        
        # Limit dataset size if specified
        if max_samples and len(self.image_files) > max_samples:
            self.image_files = self.image_files[:max_samples]

        self.filter_images_with_unvalid_text()

        print(f"Dataset loaded with {len(self.image_files)} images and {len(self.texts)} descriptions")

    def filter_images_with_unvalid_text(self):
        # drop rows in texts for image_files that are not in image_files
        self.texts = self.texts[self.texts['image_name'].isin(self.image_files)]

        for image_file in self.image_files:
            matching_texts = self.texts.loc[self.texts['image_name'] == image_file, 'text']
            if len(matching_texts) == 0:
                self.image_files.remove(image_file)

        # drop NaN from texts and corresponding image_files
        self.texts = self.texts.dropna()
        self.image_files = [f for f in self.image_files if f in self.texts['image_name'].values]

        # if text is not str, drop it to and image_file
        self.texts = self.texts[self.texts['text'].apply(lambda x: isinstance(x, str))]
        self.image_files = [f for f in self.image_files if f in self.texts['image_name'].values]


    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.image_files[idx])
        image = Image.open(img_path).convert('RGB')
        image = self.transform(image)

        # Найдем текст для изображения
        matching_texts = self.texts.loc[self.texts['image_name'] == self.image_files[idx], 'text']
        
        # If there are multiple texts for this image, randomly select one
        if len(matching_texts) > 1:
            text_idx = random.randint(0, len(matching_texts) - 1)
            matching_texts = matching_texts.iloc[[text_idx]]

        # Проверим, есть ли текст для этого изображения
        if len(matching_texts) == 0:
            # Если текста нет, вызовем ошибку
            raise ValueError(f"No text found for image {self.image_files[idx]}")

        # Если текст есть, берем первый
        text = matching_texts.values[0]

        return image, text


def visualize_batch(images, noisy_images, outputs, epoch, batch_idx, texts=None, desc="val", clearml_logger=None):
    """
    Visualize original images, noisy images, and model outputs for debugging and log to ClearML
    """
    if clearml_logger is None:
        return

    # Select a few samples
    num_samples = min(1, images.size(0))
    
    # Denormalize images
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1).to(images.device)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1).to(images.device)
    
    for i in range(num_samples):
        # Create figure with subplots
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # Original image - детач и конвертация в CPU до любых операций
        with torch.no_grad():
            orig_img = images[i].cpu().detach() * std.cpu() + mean.cpu()
            orig_img = orig_img.permute(1, 2, 0).float().numpy()
            orig_img = np.clip(orig_img, 0, 1)
            
            # noisy image
            noisy_img = noisy_images[i].cpu().detach() * std.cpu() + mean.cpu()
            noisy_img = noisy_img.permute(1, 2, 0).float().numpy()
            noisy_img = np.clip(noisy_img, 0, 1)
            
            # outputs image
            output_img = outputs[i].cpu().detach() * std.cpu() + mean.cpu()
            output_img = output_img.permute(1, 2, 0).float().numpy()
            output_img = np.clip(output_img, 0, 1)

        # Plot images
        axes[0].imshow(orig_img)
        axes[0].set_title("Original")
        axes[0].axis("off")
        
        axes[1].imshow(noisy_img)
        axes[1].set_title("Noisy (SD)")
        axes[1].axis("off")
        
        axes[2].imshow(output_img, cmap='viridis')
        axes[2].set_title("Generated")
        axes[2].axis("off")
        
        # Set title with text prompt if available
        if texts is not None:
            plt.suptitle(f"Prompt: {texts[i][:50]}...")
            
        plt.tight_layout()
        
        # Log to ClearML
        # Convert plt figure to numpy array for proper logging
        # Save figure to temporary file and load with PIL
        temp_path = f"generated_samples/temp_plot_{epoch}_{batch_idx}_{i}.png"
        plt.savefig(temp_path)
        
        try:
            image_from_plot = Image.open(temp_path).convert('RGB')  # Convert to RGB mode
            clearml_logger.get_logger().report_image(
                title=f"(Images {epoch})",
                series=f"{desc} sample {i} (batch {batch_idx})", 
                image=image_from_plot,
                iteration=epoch
            )
            # Явно закрываем изображение
            image_from_plot.close()
        except Exception as e:
            print(f"Error reporting image to ClearML: {e}")
        finally:
            # Всегда удаляем временный файл
            if os.path.exists(temp_path):
                os.remove(temp_path)  # Clean up temp file
        
        # Очистка matplotlib
        plt.close(fig)
        
    # Явно очищаем кэш CUDA
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        
def validate(model, val_loader, device, kl_weight=0.01, epoch=0, debug=False, max_batches=None, clearml_logger=None):
    """
    Проверяет модель на валидационном наборе данных.
    
    Args:
        model: модель для проверки
        val_loader: даталоадер с валидационными данными
        device: устройство для вычислений ('cuda' или 'cpu')
        kl_weight: вес KL-дивергенции в функции потерь
        debug: флаг отладочного режима
        max_batches: максимальное количество батчей для валидации в отладочном режиме
    
    Returns:
        Кортеж со средними значениями всех метрик (loss, mse_loss, kl_loss, recon_loss)
    """
    # Переводим модель в режим оценки
    model.eval()
    
    # Инициализируем счетчики для метрик
    total_loss = 0
    total_mse_loss = 0
    total_kl_loss = 0
    total_recon_loss = 0
    batch_count = 0
    
    # Отключаем вычисление градиентов для ускорения и экономии памяти
    with torch.no_grad():
        progress_bar = tqdm(val_loader, desc='Validation')
        
        for batch_idx, (images, texts) in enumerate(progress_bar):
            images = images.to(device)
            
            # Создаем зашумленные изображения аналогично функции train
            if True:
                # В режиме отладки используем быстрый метод генерации шума
                noisy_images = diffuse_silhouette_effect_strong(
                    images, noise_type='gaussian', blur_ksize=61, noise_strength=0.4
                )
                noisy_images = noisy_images.to(device)
            else:
                # Генерируем зашумленные изображения с помощью Stable Diffusion 1.5
                noisy_images = []
                for text in texts:
                    noisy_img = model.generate_noisy_image(text, num_inference_steps=3 if debug else 10)
                    noisy_images.append(noisy_img)
                noisy_images = torch.cat(noisy_images, dim=0).to(device)
            
            # Прямой проход через модель с заданным весом KL-дивергенции
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
                loss, outputs = model(texts, images, noisy_images, kl_weight=kl_weight)
            
            # Явный детач для всех выходных тензоров
            outputs = {k: v.detach() if isinstance(v, torch.Tensor) else v for k, v in outputs.items()}
            
            # Извлекаем компоненты функции потерь
            mse_loss = outputs['mse_loss'].item()  # Используем .item() вместо хранения тензора
            kl_loss = outputs['kl_loss'].item()
            recon_loss = outputs['reconstruction_loss'].item()
            loss_value = loss.item()
            
            # Обновляем метрики (используем числа, а не тензоры)
            total_loss += loss_value
            total_mse_loss += mse_loss
            total_kl_loss += kl_loss
            total_recon_loss += recon_loss
            batch_count += 1
            
            # Отображаем текущие значения метрик
            progress_bar.set_postfix({
                'val_loss': total_loss / batch_count,
                'val_mse': total_mse_loss / batch_count,
                'val_kl': total_kl_loss / batch_count,
                'val_recon': total_recon_loss / batch_count
            })

            if batch_idx % 10 == 0:
                visualize_batch(
                    images, 
                    noisy_images, 
                    outputs['predicted_image'], 
                    epoch+1, 
                    batch_idx, 
                    desc="val",
                    texts=texts, 
                    clearml_logger=clearml_logger
                )
            
            # Очищаем ненужные переменные
            del images, noisy_images, outputs, loss
            
        # Очищаем кэш CUDA в конце валидации
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    # Вычисляем средние значения метрик
    avg_loss = total_loss / batch_count if batch_count > 0 else 0
    avg_mse_loss = total_mse_loss / batch_count if batch_count > 0 else 0
    avg_kl_loss = total_kl_loss / batch_count if batch_count > 0 else 0
    avg_recon_loss = total_recon_loss / batch_count if batch_count > 0 else 0
    
    # Выводим результаты валидации
    print(f'Validation - Loss: {avg_loss:.4f}, MSE: {avg_mse_loss:.4f}, KL: {avg_kl_loss:.4f}, Recon: {avg_recon_loss:.4f}')
    
    # Возвращаем средние значения метрик
    return avg_loss, avg_mse_loss, avg_kl_loss, avg_recon_loss


def train(model, train_loader, val_loader, optimizer, device, num_epochs=10, save_dir="models", 
          save_interval=5, debug=False, max_batches_per_epoch=None,
          latent_dim=256, kl_weight=0.01, use_clearml=False, warmup_epochs=5,
          gradient_clip_value=1.0, start_epoch=0):
    """
    Train the model with optional debug mode
    
    Args:
        model: модель для обучения
        train_loader: даталоадер с обучающими данными
        val_loader: даталоадер с валидационными данными
        optimizer: оптимизатор
        device: устройство для вычислений
        num_epochs: общее количество эпох
        save_dir: директория для сохранения чекпоинтов
        save_interval: интервал сохранения чекпоинтов (в эпохах)
        debug: флаг отладочного режима
        max_batches_per_epoch: максимальное число батчей в эпохе (для отладки)
        latent_dim: размерность латентного пространства
        kl_weight: вес KL-дивергенции в функции потерь
        use_clearml: флаг использования ClearML для логирования
        warmup_epochs: количество эпох разогрева для планировщика скорости обучения
        gradient_clip_value: максимальная норма градиента для отсечения
        start_epoch: начальная эпоха (для продолжения обучения)
    """
    # Create directory for saving models
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs("generated_samples", exist_ok=True)
    
    # Initialize TensorBoard writer
    writer = SummaryWriter(log_dir=f'runs/omni-gpt2o{"debug_" if debug else ""}latent{latent_dim}')
    
    # Инициализируем ClearML logger если он используется
    clearml_logger = None
    if use_clearml and Task is not None:
        clearml_logger = Task.current_task()
        if clearml_logger:
            print("Using ClearML for experiment tracking")
    
    # Инициализируем планировщик обучения с косинусным затуханием и разогревом
    total_steps = len(train_loader) * num_epochs
    warmup_steps = len(train_loader) * warmup_epochs
    
    def cosine_schedule_with_warmup(step):
        if step < warmup_steps:
            # Линейное увеличение в период разогрева
            return float(step) / float(max(1, warmup_steps))
        else:
            # Косинусное затухание после разогрева
            progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
            return max(0.0, 0.5 * (1.0 + np.cos(np.pi * progress)))
    
    # Создаем планировщик обучения
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=cosine_schedule_with_warmup)
    
    # Если мы продолжаем обучение, нужно правильно настроить планировщик
    if start_epoch > 0:
        # Делаем шаги планировщика, чтобы достичь правильного значения LR
        for _ in range(start_epoch * len(train_loader)):
            scheduler.step()
    
    model.train()
    best_loss = float('inf')
    
    for epoch in range(start_epoch, num_epochs):
        epoch_start_time = time.time()
        total_loss = 0
        total_mse_loss = 0
        total_kl_loss = 0
        total_recon_loss = 0
        total_grad_norm = 0.0
        progress_bar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs} [Train]')
        
        for batch_idx, (images, texts) in enumerate(progress_bar):
            images = images.detach().to(device)
            
            # Generate noisy images with Stable Diffusion 1.5
            if True:
                # In debug mode, use simpler noise generation for speed
                # Add strong Perlin-like noise to simulate early diffusion steps
                noisy_images = diffuse_silhouette_effect_strong(
                    images, noise_type='gaussian', blur_ksize=61, noise_strength=0.4
                )
                noisy_images = noisy_images.detach().to(device)
            else:  # TODO: preprocess dataset before training with SD 1.5
                # Generate actual noisy images with SD 1.5
                noisy_images = []
                for text in texts:
                    noisy_img = model.generate_noisy_image(text, num_inference_steps=3 if debug else 10)
                    noisy_images.append(noisy_img)
                noisy_images = torch.cat(noisy_images, dim=0).to(device)
            
            # Forward pass with specified KL weight
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
                loss, outputs = model(texts, images, noisy_images, kl_weight=kl_weight)
            
            # Extract individual loss components (используем .item() для скалярных значений)
            mse_loss_value = outputs['mse_loss'].item()
            kl_loss_value = outputs['kl_loss'].item()
            recon_loss_value = outputs['reconstruction_loss'].item()
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            
            # Вычисляем норму градиента для каждого параметра
            grad_norm = 0.0
            for param in model.parameters():
                if param.grad is not None:
                    param_norm = param.grad.data.norm(2)
                    grad_norm += param_norm.item() ** 2
            grad_norm = grad_norm ** 0.5
            total_grad_norm += grad_norm
            
            # Применяем gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=gradient_clip_value)
            
            # Делаем шаг оптимизатора
            optimizer.step()
            
            # Обновляем learning rate
            scheduler.step()
            
            # Логируем текущий learning rate и градиенты
            current_lr = scheduler.get_last_lr()[0]
            if batch_idx % 10 == 0:
                writer.add_scalar('LearningRate', current_lr, epoch * len(train_loader) + batch_idx)
                writer.add_scalar('Gradients/Norm', grad_norm, epoch * len(train_loader) + batch_idx)
                
                # Логируем градиенты отдельных компонентов модели
                # for name, param in model.named_parameters():
                #     if param.requires_grad and param.grad is not None:
                #         writer.add_histogram(f'Gradients/{name}', param.grad, epoch * len(train_loader) + batch_idx)
                
                if clearml_logger:
                    clearml_logger.get_logger().report_scalar(
                        title="Training", series="Learning Rate", 
                        value=current_lr, iteration=epoch * len(train_loader) + batch_idx)
                    clearml_logger.get_logger().report_scalar(
                        title="Gradients", series="Gradient Norm", 
                        value=grad_norm, iteration=epoch * len(train_loader) + batch_idx)
            
            # Update metrics using scalar values
            total_loss += loss.item()
            total_mse_loss += mse_loss_value
            total_kl_loss += kl_loss_value
            total_recon_loss += recon_loss_value
            
            progress_bar.set_postfix({
                'loss': total_loss / (batch_idx + 1),
                'mse': total_mse_loss / (batch_idx + 1),
                'kl': total_kl_loss / (batch_idx + 1),
                'recon': total_recon_loss / (batch_idx + 1),
                'lr': current_lr,
                'grad': grad_norm
            })
            
            # Очищаем переменные для высвобождения памяти
            # del images, noisy_images, loss
            # Сохраняем outputs['predicted_image'] для visualize_batch ниже
            predicted_image = outputs['predicted_image'].detach()
            del outputs

        # Visualize only in certain batches and end of epoch
        visualize_batch(
            images, 
            noisy_images, 
            predicted_image, 
            epoch+1, 
            batch_idx, 
            desc="train",
            texts=texts, 
            clearml_logger=clearml_logger
        )

        del images, noisy_images, loss

        # Очищаем оставшиеся переменные
        if 'predicted_image' in locals():
            del predicted_image
        if 'texts' in locals():
            del texts
        
        # Очистка CUDA кэша
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
        # Calculate epoch stats
        num_batches = batch_idx + 1
        avg_loss = total_loss / num_batches
        avg_mse_loss = total_mse_loss / num_batches
        avg_kl_loss = total_kl_loss / num_batches
        avg_recon_loss = total_recon_loss / num_batches
        epoch_time = time.time() - epoch_start_time
        
        print(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {avg_loss:.4f}, MSE: {avg_mse_loss:.4f}, KL: {avg_kl_loss:.4f}, Recon: {avg_recon_loss:.4f}, Time: {epoch_time:.2f}s, LR: {current_lr:.6f}')
        
        # Log to TensorBoard
        writer.add_scalar('Loss/train_total', avg_loss, epoch)
        writer.add_scalar('Loss/train_mse', avg_mse_loss, epoch)
        writer.add_scalar('Loss/train_kl', avg_kl_loss, epoch)
        writer.add_scalar('Loss/train_reconstruction', avg_recon_loss, epoch)

        # Validation
        val_metrics = (0, 0, 0, 0)  # Default values if no validation
        if val_loader is not None:
            val_metrics = validate(
                model, val_loader, device, 
                kl_weight=kl_weight, epoch=epoch, debug=debug, 
                max_batches=max_batches_per_epoch, clearml_logger=clearml_logger
            )
            avg_val_loss, avg_val_mse_loss, avg_val_kl_loss, avg_val_recon_loss = val_metrics
            
            # Log validation metrics to TensorBoard
            writer.add_scalar('Loss/val_total', avg_val_loss, epoch)
            writer.add_scalar('Loss/val_mse', avg_val_mse_loss, epoch)
            writer.add_scalar('Loss/val_kl', avg_val_kl_loss, epoch)
            writer.add_scalar('Loss/val_reconstruction', avg_val_recon_loss, epoch)

        # Log to ClearML
        if clearml_logger:
            clearml_logger.get_logger().report_scalar(
                title="Loss", series="Train Total", value=avg_loss, iteration=epoch)
            clearml_logger.get_logger().report_scalar(
                title="Loss", series="Train MSE", value=avg_mse_loss, iteration=epoch)
            clearml_logger.get_logger().report_scalar(
                title="Loss", series="Train KL", value=avg_kl_loss, iteration=epoch)
            clearml_logger.get_logger().report_scalar(
                title="Loss", series="Train Reconstruction", value=avg_recon_loss, iteration=epoch)
            
            # Log validation metrics to ClearML if available
            if val_loader is not None:
                clearml_logger.get_logger().report_scalar(
                    title="Loss", series="Val Total", value=avg_val_loss, iteration=epoch)
                clearml_logger.get_logger().report_scalar(
                    title="Loss", series="Val MSE", value=avg_val_mse_loss, iteration=epoch)
                clearml_logger.get_logger().report_scalar(
                    title="Loss", series="Val KL", value=avg_val_kl_loss, iteration=epoch)
                clearml_logger.get_logger().report_scalar(
                    title="Loss", series="Val Reconstruction", value=avg_val_recon_loss, iteration=epoch)

        # Determine which loss to use for model saving (validation if available, otherwise training)
        evaluation_loss = avg_val_loss if val_loader is not None else avg_loss

        # Save model checkpoint
        if (epoch + 1) % save_interval == 0 or epoch == num_epochs - 1 or debug:
            checkpoint_path = f"{save_dir}/omni-gpt2o{'debug_' if debug else ''}latent{latent_dim}_last.pt"
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': evaluation_loss,
                'latent_dim': latent_dim,
            }, checkpoint_path)
            print(f"Model saved to {checkpoint_path}")
            
            # Save best model
            if evaluation_loss < best_loss:
                best_loss = evaluation_loss
                best_checkpoint_path = f"{save_dir}/omni-gpt2o{'debug_' if debug else ''}latent{latent_dim}_best.pt"
                torch.save({
                    'epoch': epoch + 1,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': evaluation_loss,
                    'latent_dim': latent_dim,
                }, best_checkpoint_path)
                print(f"Best model saved to {best_checkpoint_path} (loss: {evaluation_loss:.4f})")

    
    writer.close()


def load_checkpoint(model, checkpoint_path, device, latent_dim, optimizer=None):
    start_epoch = 0
    if checkpoint_path is not None:
        if os.path.isfile(checkpoint_path):
            print(f"Loading checkpoint from {checkpoint_path}")
            try:
                # Сначала пробуем загрузить с weights_only=False для безопасности
                checkpoint = torch.load(checkpoint_path, map_location=device)
                
                # Загружаем веса модели выборочно
                model_state = model.state_dict()
                checkpoint_state = checkpoint['model_state_dict']
                
                # Загружаем только те веса, которые совпадают по размерам
                for name, param in checkpoint_state.items():
                    if name in model_state:
                        try:
                            if model_state[name].shape == param.shape:
                                model_state[name].copy_(param)
                                print(f"Loaded layer: {name}")
                            else:
                                print(f"Skipped layer due to shape mismatch: {name}")
                        except Exception as e:
                            print(f"Error loading layer {name}: {str(e)}")
                
                try:
                    # Пробуем загрузить состояние оптимизатора
                    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                    print("Loaded optimizer state")
                except Exception as e:
                    print(f"Could not load optimizer state: {str(e)}")
                
                # Получаем информацию о чекпоинте
                start_epoch = checkpoint.get('epoch', 0)
                print(f"Loaded checkpoint from epoch {start_epoch}, continuing training...")
                
                # Проверяем соответствие latent_dim
                checkpoint_latent_dim = checkpoint.get('latent_dim', latent_dim)
                if checkpoint_latent_dim != latent_dim:
                    print(f"WARNING: Checkpoint latent_dim ({checkpoint_latent_dim}) differs from "
                          f"provided latent_dim ({latent_dim})!")
                
                # Освобождаем память
                del checkpoint
                torch.cuda.empty_cache()
                
            except Exception as e:
                print(f"Error loading checkpoint: {str(e)}")
                print("Starting from scratch")
                start_epoch = 0
        else:
            print(f"WARNING: Checkpoint file not found at {checkpoint_path}, starting from scratch.")

    return start_epoch

def main():
    # Parse arguments
    parser = argparse.ArgumentParser(description="Train the ViT-VAE model")
    parser.add_argument("--debug", action="store_true", help="Run in debug mode with limited data")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    parser.add_argument("--epochs", type=int, default=100, help="Number of epochs")
    parser.add_argument("--max_train_samples", type=int, default=20000, help="Maximum number of samples to use")
    parser.add_argument("--max_val_samples", type=int, default=200, help="Maximum number of samples to use")
    parser.add_argument("--save_interval", type=int, default=5, help="Save checkpoint every N epochs")
    parser.add_argument("--latent_dim", type=int, default=768, help="Dimension of latent space")
    parser.add_argument("--kl_weight", type=float, default=0.01, help="Weight for KL divergence loss")
    parser.add_argument("--use_clearml", action="store_true", help="Use ClearML for experiment tracking")
    parser.add_argument("--warmup_epochs", type=int, default=3, help="Number of epochs for warmup")
    parser.add_argument("--gradient_clip", type=float, default=1.0, help="Gradient clipping value")
    parser.add_argument("--learning_rate", "--lr", type=float, default=1e-4, help="Initial learning rate")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--checkpoint_path", type=str, default=None, 
                        help="Path to checkpoint file to resume training from")
    args = parser.parse_args()

    # Устанавливаем seed для воспроизводимости
    set_seed(args.seed)

    # Инициализируем ClearML
    if args.use_clearml and Task is not None:
        print("Initializing ClearML tracking")
        task_name = f"omni-gpt2o_latent{args.latent_dim}_{'debug' if args.debug else 'train'}_2"
        task = Task.init(project_name="Other/omni-gpt2o", task_name=task_name, 
                         auto_connect_frameworks={'pytorch': False})
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Set parameters based on debug mode
    if args.debug:
        print("Running in DEBUG mode with reduced parameters")
        batch_size = 2
        num_epochs = 3
        max_batches_per_epoch = 20
        args.max_train_samples = 200
        args.max_val_samples = 200
    else:
        batch_size = args.batch_size
        num_epochs = args.epochs
        max_batches_per_epoch = None
    
    # Initialize model with specified latent dimension
    print(f"Initializing model with latent_dim={args.latent_dim}...")
    model = ImageTextGenerator(device=device, latent_dim=args.latent_dim)
    model = model.to(device)
    
    # Initialize optimizer (only for trainable parameters)
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=args.learning_rate)
    
    # Загрузка чекпоинта при необходимости
    start_epoch = load_checkpoint(model, args.checkpoint_path, device, args.latent_dim, optimizer=optimizer)
    
    print(f"Training with learning rate: {args.learning_rate}")
    
    # Create dataset and dataloader
    print("Loading dataset...")
    train_dataset = ImageTextDataset(
        image_dir="data/prepared_dataset/train2017/images",
        text_file="data/prepared_dataset/train2017/descriptions.txt",
        max_samples=args.max_train_samples
    )
    val_dataset = ImageTextDataset(
        image_dir="data/prepared_dataset/val2017/images",
        text_file="data/prepared_dataset/val2017/descriptions.txt",
        max_samples=args.max_val_samples
    )
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=2 if args.debug else 4,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=2 if args.debug else 4,
        pin_memory=True
    )
    
    # Train the model
    print("Starting training...")
    train(
        model, 
        train_loader, 
        val_loader,
        optimizer, 
        device, 
        num_epochs=num_epochs,
        save_dir="models",
        save_interval=args.save_interval,
        debug=args.debug,
        max_batches_per_epoch=max_batches_per_epoch,
        latent_dim=args.latent_dim,
        kl_weight=args.kl_weight,
        use_clearml=args.use_clearml,
        warmup_epochs=args.warmup_epochs,
        gradient_clip_value=args.gradient_clip,
        start_epoch=start_epoch
    )
    
    print("Training completed!")

if __name__ == "__main__":
    main() 