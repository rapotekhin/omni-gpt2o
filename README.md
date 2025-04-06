# GPT-2 + ViT Image Generation Research (Omni-GPT2o)

This project implements a research approach to image generation using GPT-2 and ViT (Vision Transformer) models, inspired by recent developments in native image generation capabilities of LLMs.

## Project Structure

```
.
├── src/
│   ├── model.py      # Main model architecture
│   └── train.py      # Training script
├── utils/
│   ├── preprocess.py           # Data preprocessing utilities
│   ├── extract_descriptions.py # Script to extract COCO descriptions
│   └── noise.py                # Functions for noise generation
├── data/             # Dataset directory
├── models/           # Saved model checkpoints
├── logs/             # Training logs and visualizations
├── generate_sample.py # Image generation script
└── requirements.txt  # Project dependencies
```

## Setup

1. Install dependencies:
```bash
python -m venv venv
source venv/bin/activate  # On Linux/Mac
# or
venv\Scripts\activate     # On Windows
pip install -r requirements.txt
```

2. Download and prepare your dataset:

### Download COCO dataset
```bash
# Create data directories
mkdir -p data/prepared_dataset/train2017/images
mkdir -p data/prepared_dataset/val2017/images

# Download COCO annotations and images
wget http://images.cocodataset.org/annotations/annotations_trainval2017.zip -P data/
wget http://images.cocodataset.org/zips/train2017.zip -P data/
wget http://images.cocodataset.org/zips/val2017.zip -P data/
```

### Extract archives
```bash
# Extract annotations
unzip -q data/annotations_trainval2017.zip -d data/

# Extract images
unzip -q data/train2017.zip -d data/
unzip -q data/val2017.zip -d data/

# Move images to their respective directories
mv data/train2017/* data/prepared_dataset/train2017/images/
mv data/val2017/* data/prepared_dataset/val2017/images/
```

### Process the dataset
```bash
# Extract image descriptions for training set
python utils/extract_descriptions.py \
    --annotations_file data/annotations/captions_train2017.json \
    --output_file data/prepared_dataset/train2017/descriptions.txt

# Extract image descriptions for validation set
python utils/extract_descriptions.py \
    --annotations_file data/annotations/captions_val2017.json \
    --output_file data/prepared_dataset/val2017/descriptions.txt

```

3. Train the model:
```bash
# Run training with default parameters
python src/train.py --epochs 100 --batch_size 16 --latent_dim 256

# Run with custom parameters
python src/train.py \
    --epochs 100 \
    --batch_size 16 \
    --latent_dim 512 \
    --learning_rate 2e-4 \
    --warmup_epochs 10 \
    --gradient_clip 1.0 \
    --kl_weight 0.05 \
    --max_train_samples 50000 \
    --max_val_samples 1000 \
    --seed 42
    
# Run in debug mode (faster with less data)
python src/train.py --debug
```

4. Generate images from text prompts:
```bash
python generate_sample.py --prompt "A cat wearing a space suit on Mars" --model models/omni-gpt2o_latent256_best.pt
```

## Model Architecture

The system consists of multiple components:

### Text-to-Image Generation
- GPT-2 for text processing
- ViT (Vision Transformer) for image encoding
- Stable Diffusion for generating noisy images
- A linear projection layer to match dimensions between ViT and GPT-2

### Image Generation Pipeline
1. Encode text prompts with GPT-2
2. Generate noisy images with Stable Diffusion
3. Encode both text and noisy images
4. Project to latent space
5. Decode using the decoder to create final images

## Requirements

- Python 3.8+
- PyTorch 2.0+
- Transformers 4.30+
- Diffusers 0.18+
- Timm 0.9.0+
- Matplotlib
- TensorBoard
- Optional: ClearML for experiment tracking
- Other dependencies listed in requirements.txt

## Notes

- The GPT-2 encoder and ViT models are frozen during training
- Only projection layers and decoders are trained
- This research explores an alternative to DALLE-like systems using established models 