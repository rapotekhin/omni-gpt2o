# GPT-2 + ViT Image Generation Research

This project implements a research approach to image generation using GPT-2 and ViT (Vision Transformer) models, inspired by recent developments in native image generation capabilities of LLMs.

## Project Structure

```
.
├── src/
│   ├── model.py      # Main model architecture
│   └── train.py      # Training script
├── utils/
│   └── preprocess.py # Data preprocessing utilities
├── data/             # Dataset directory
├── models/           # Saved model checkpoints
└── requirements.txt  # Project dependencies
```

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Prepare your dataset:
```bash
python utils/preprocess.py --image_dir /path/to/images --output_dir data --descriptions_file /path/to/descriptions.txt
```

3. Start training:
```bash
python src/train.py
```

## Model Architecture

The model combines several components:
- GPT-2 for text processing
- ViT (Vision Transformer) for image encoding/decoding
- Stable Diffusion for generating noisy images
- A linear projection layer to match dimensions between ViT and GPT-2

The training process involves:
1. Encoding text prompts with GPT-2
2. Generating noisy images with Stable Diffusion
3. Encoding both original and noisy images with ViT
4. Training the model to predict original image tokens from text and noisy image embeddings

## Requirements

- Python 3.8+
- PyTorch 2.0+
- Transformers 4.30+
- Diffusers 0.18+
- Other dependencies listed in requirements.txt

## Notes

- The GPT-2 encoder and ViT models are frozen during training
- Only the linear projection layer and GPT-2 decoder are trained
- Stable Diffusion is used with minimal steps (3) to generate noisy images
- The model learns to translate between noisy and clean image representations 