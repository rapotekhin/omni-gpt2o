import torch
import numpy as np
from PIL import Image
from torchvision import transforms
import cv2

def make_perlin_noise(image, noise_level=0.8):
    """
    Создает шум для изображения, сохраняя его структуру.
    Шум применяется с учетом формы и структуры исходного изображения.
    
    Args:
        image: тензор изображения в формате (C, H, W)
        noise_level: сила шума от 0 до 1
    
    Returns:
        Зашумленное изображение в том же формате, что и входное
    """
    # Создаем шум такой же формы как исходное изображение
    noise = torch.randn_like(image)
    
    # Вычисляем маску структуры на основе градиентов исходного изображения
    # Это позволяет сохранить более важные структурные элементы
    gray = (0.2989 * image[0] + 0.5870 * image[1] + 0.1140 * image[2]).unsqueeze(0)
    structure_mask = 1.0 - torch.abs(gray - gray.mean()) / (gray.std() + 1e-5)
    structure_mask = structure_mask.repeat(3, 1, 1)
    
    # Применяем шум с учетом структуры изображения
    # В местах со значимой структурой (низкое значение маски) применяется меньше шума
    adaptive_strength = noise_level * structure_mask
    noisy_image = image * (1 - adaptive_strength) + noise * adaptive_strength
    
    # Ограничиваем значения в диапазоне [-1, 1] для корректного отображения
    return torch.clamp(noisy_image, -1, 1)

def make_gaussian_blur(image, blur_radius=5, noisse_level=0.3):
    """
    Применяет гауссово размытие к изображению и добавляет шум.
    
    Args:
        image: тензор изображения в формате (C, H, W)
        blur_radius: радиус размытия (нечетное число)
        noise_level: сила шума от 0 до 1
    
    Returns:
        Размытое изображение с шумом в формате (C, H, W)
    """
    # Преобразуем тензор в numpy массив для обработки в OpenCV
    img_np = image.cpu().numpy().transpose(1, 2, 0)  # (C,H,W) -> (H,W,C)
    
    # Ограничиваем значения в диапазоне [0, 1] для корректного отображения
    img_np = (img_np - img_np.min()) / (img_np.max() - img_np.min())
    
    # Убеждаемся, что радиус размытия нечетный
    if blur_radius % 2 == 0:
        blur_radius += 1
    
    # Применяем гауссово размытие
    blurred = cv2.GaussianBlur(img_np, (blur_radius, blur_radius), 0)
    
    noisy_image = blurred
    # Ограничиваем значения в диапазоне [0, 1] для корректного отображения
    noisy_image = np.clip(noisy_image, 0, 1)
    
    # Преобразуем обратно в тензор
    return torch.from_numpy(noisy_image.transpose(2, 0, 1)).float()

def diffuse_silhouette_effect_strong(image, noise_type='gaussian', blur_ksize=51, noise_strength=0.6):
    """
    Применяет силуэтный эффект к изображению с сильным размытием.
    
    Args:
        image: тензор изображения в формате (B, C, H, W) или (C, H, W)
        noise_type: тип шума ('gaussian', 'uniform', 'salt_pepper')
        blur_ksize: размер ядра размытия (нечетное число)
        noise_strength: сила шума
    
    Returns:
        Зашумленное изображение с силуэтным эффектом в том же формате, что и входное
    """
    # Убеждаемся что работаем с тензором
    numpy_input = False
    if not isinstance(image, torch.Tensor):
        numpy_input = True
        image = torch.from_numpy(image)
    
    # Проверяем размерность входного тензора
    is_batched = len(image.shape) == 4  # [B, C, H, W]
    batch_size = image.shape[0] if is_batched else 1
    
    # Если входное изображение не в батче, добавляем размерность батча
    if not is_batched:
        image = image.unsqueeze(0)  # [1, C, H, W]
    
    # Получаем размеры изображения
    _, channels, height, width = image.shape
    
    # Создаем выходной тензор
    result = []
    
    # Обрабатываем каждое изображение в батче
    for b in range(batch_size):
        # Создаем список для хранения обработанных каналов текущего изображения
        img_channels = []
        
        # Обрабатываем каждый канал отдельно
        for c in range(channels):
            channel = image[b, c]  # [H, W]
            
            # Применяем шум
            if noise_type == 'gaussian':
                noise = torch.randn_like(channel) * noise_strength
            elif noise_type == 'uniform':
                noise = (torch.rand_like(channel) * 2 - 1) * noise_strength
            elif noise_type == 'salt_pepper':
                noise = torch.zeros_like(channel)
                prob = noise_strength
                salt = torch.rand_like(channel) < prob / 2
                pepper = torch.rand_like(channel) > 1 - prob / 2
                noise[salt] = 1
                noise[pepper] = -1
            else:
                raise ValueError("Unsupported noise type")
                
            noisy_image = torch.clamp(channel + noise, 0, 1)
            
            # Убеждаемся, что размер ядра размытия нечетный
            if blur_ksize % 2 == 0:
                blur_ksize += 1
                
            # Создаем гауссово ядро
            sigma = blur_ksize / 6.0
            kernel_size = blur_ksize
            kernel_x = torch.arange(kernel_size, dtype=torch.float32) - kernel_size // 2
            kernel = torch.exp(-0.5 * (kernel_x / sigma) ** 2)
            kernel = kernel / kernel.sum()
            
            # Применяем размытие с OpenCV для упрощения
            noisy_np = noisy_image.cpu().numpy()
            blurred_np = cv2.GaussianBlur(noisy_np, (blur_ksize, blur_ksize), sigma)
            blurred = torch.from_numpy(blurred_np).to(image.device)
            
            # Нормализация
            if blurred.max() > blurred.min():  # Избегаем деления на ноль
                final_image = (blurred - blurred.min()) / (blurred.max() - blurred.min())
            else:
                final_image = blurred
                
            img_channels.append(final_image)
            
        # Объединяем каналы текущего изображения
        img_result = torch.stack(img_channels)  # [C, H, W]
        result.append(img_result)
    
    # Объединяем результаты всех изображений в батче
    result = torch.stack(result)  # [B, C, H, W]
    
    # Если входное изображение не было в батче, убираем размерность батча
    if not is_batched:
        result = result.squeeze(0)  # [C, H, W]

    if numpy_input:
        return result.cpu().numpy()
    
    return result


def change_noise_func(image, args):
    if args.get("noise_func") == "make_perlin_noise":
        return make_perlin_noise(image, args.get("noise_level", 0.8))
    elif args.get("noise_func") == "diffuse_silhouette_effect_strong":
        return diffuse_silhouette_effect_strong(
            image, 
            args.get("noise_type", "gaussian"), 
            args.get("blur_ksize", 71),
            args.get("noise_strength", 0.6)
        )
    else:
        return make_gaussian_blur(image, args.get("blur_radius", 5), args.get("noise_level", 0.3))

if __name__ == "__main__":
    # Тестирование функции на конкретном изображении
    image_path = "/mnt/sda/omni-gpt2o/data/prepared_dataset/val2017/images/000000179765.jpg"
    image = Image.open(image_path).convert('RGB')

    # Преобразуем в тензор для обработки
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    image_tensor = transform(image)
    
    # Создаем и сохраняем оригинальное изображение
    orig_image = image_tensor.cpu().numpy()
    orig_image = np.transpose(orig_image, (1, 2, 0))
    orig_image = (orig_image * 255).astype(np.uint8)
    Image.fromarray(orig_image).save("000000179765_original.png")

    # Тестируем все типы шума
    noise_types = [
        {"noise_func": "make_perlin_noise", "noise_level": 0.7},
        {"noise_func": "make_gaussian_blur", "blur_radius": 11, "noise_level": 0.3},
        {"noise_func": "diffuse_silhouette_effect_strong", "noise_type": "gaussian", "blur_ksize": 61, "noise_strength": 0.4}
    ]
    
    for i, args in enumerate(noise_types):
        # Создаем и сохраняем зашумленное изображение
        noisy_image = change_noise_func(image_tensor, args)
        noisy_image = noisy_image.cpu().numpy()
        # Преобразуем (C, H, W) в (H, W, C) для PIL
        noisy_image = np.transpose(noisy_image, (1, 2, 0))
        # Нормализуем в диапазон [0, 1]
        noisy_image = (noisy_image - noisy_image.min()) / (noisy_image.max() - noisy_image.min())
        # Преобразуем в 8-битное изображение
        noisy_image = (noisy_image * 255).astype(np.uint8)
        Image.fromarray(noisy_image).save(f"000000179765_noisy_{args['noise_func']}.png")
    
    # Создаем сравнительное изображение для функции силуэтов с разными порогами
    thresholds = [0.4, 0.5, 0.6, 0.7]
    fig_width = 224 * (len(thresholds) + 1)
    fig_height = 224
    
    silhouette_comparison = Image.new('RGB', (fig_width, fig_height))
    silhouette_comparison.paste(Image.fromarray(orig_image), (0, 0))
    
    for i, threshold in enumerate(thresholds):
        args = {"noise_func": "diffuse_silhouette_effect_strong", "noise_type": "gaussian", "blur_ksize": 51, "noise_strength": threshold}
        silhouette = change_noise_func(image_tensor, args)
        silhouette_np = silhouette.cpu().numpy()
        silhouette_np = np.transpose(silhouette_np, (1, 2, 0))
        silhouette_np = (silhouette_np * 255).astype(np.uint8)
        silhouette_img = Image.fromarray(silhouette_np)
        silhouette_comparison.paste(silhouette_img, ((i+1)*224, 0))
    
    silhouette_comparison.save("000000179765_silhouette_comparison.png")
    
    print("Изображения сохранены:")
    print("1. 000000179765_original.png - исходное изображение")
    print("2. 000000179765_noisy_make_perlin_noise.png - зашумленное изображение (Perlin, сила шума = 0.7)")
    print("3. 000000179765_noisy_make_gaussian_blur.png - размытое изображение с шумом (Gaussian, радиус = 11, шум = 0.3)")
    print("4. 000000179765_noisy_make_silhouette.png - силуэты объектов (порог = 0.3)")
    print("5. 000000179765_silhouette_comparison.png - сравнение силуэтов с разными порогами (0.1-0.5)")
