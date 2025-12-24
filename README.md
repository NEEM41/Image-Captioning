# Image Captioning with CLIP and a GPT-style Decoder

This project implements an image captioning model that leverages the power of OpenAI's CLIP for image understanding and a GPT-style decoder for text generation. The model is built using PyTorch and PyTorch Lightning for streamlined training and scalability.

## Overview

The core idea is to use a pre-trained CLIP model to extract rich visual features from images. These image embeddings are then used as a prefix to a decoder-only transformer model (similar to GPT), which autoregressively generates a descriptive caption.

## Features
- **CLIP-conditioned text generation:** Uses CLIP image embeddings as a prefix to guide caption generation.
- **GPT-style Decoder:** A custom transformer decoder based on the nanoGPT architecture for efficient and effective text generation.
- **PyTorch Lightning:** For clean, organized, and scalable model training.
- **MLFlow Integration:** For logging and tracking experiments.
- **Config-driven:** All hyperparameters and paths are managed through a single YAML configuration file.
- **Inference Script:** Includes a script to perform inference, generate captions for a dataset, and evaluate using BLEU scores.

## Model Architecture

The main model, `ClipLM`, consists of:
1.  **CLIP Projection:** A linear projection layer that maps the fixed-size CLIP image embedding into a sequence of prefix embeddings.
2.  **Transformer Decoder:** A stack of transformer blocks that process the prefix embeddings and the tokenized caption. It uses causal self-attention to ensure that caption generation is autoregressive.
3.  **Language Model Head:** A final linear layer that outputs logits over the vocabulary for next-token prediction.

The entire model is wrapped in a `ClipLMLightning` module for training with PyTorch Lightning.

## Setup

1.  **Clone the repository:**
    git clone <your-repo-url>
    cd Image-Captioning
    2.  **Install dependencies:**
    You will need to have `torch`, `lightning`, `pyyaml`, `sacrebleu`, `tqdm`, and `mlflow` installed.

3.  **Dataset:**
    The model expects the data to be in CSV files, with columns for image paths/IDs and tokenized captions. You should have separate CSVs for training and validation.

4.  **CLIP Embeddings:**
    This model uses pre-computed CLIP embeddings. You'll need to generate these for your image dataset and place them in a directory. The path to this directory is specified in the configuration file (`embeddings_root`).

5.  **Tokenizer:**
    The project uses a custom BPE tokenizer. You will need to train a tokenizer on your caption data and have the tokenizer model file available.

## How to Run

### Training

The model is trained using the `train.py` script, which takes a configuration file as an argument.

1.  **Configure your run:**
    Open `config/base_config_simple_transformer.yaml` and adjust the parameters, especially the paths to your dataset and embeddings.

2.  **Start training:**
    python train.py --config config/base_config_simple_transformer.yaml
        Training logs and checkpoints will be saved in the directory specified by `output_dir` in your config file, organized by `MLFlow`.

### Inference

To generate captions for a dataset split (e.g., validation set), use the `greedy_predict.py` script.

python greedy_predict.py \
    --config /path/to/your/config.yaml \
    --ckpt /path/to/your/model.ckpt \
    --out_csv predictions.csv \
    --tokenizer_model /path/to/your/tokenizer.model**Arguments:**
- `--config`: Path to the same config file used for training.
- `--ckpt`: Path to the trained model checkpoint.
- `--out_csv`: Path to save the output CSV file with predictions and BLEU scores.
- `--tokenizer_model`: Path to the BPE tokenizer model file.
- `--split`: (Optional) `train` or `val`. Defaults to `val`.
- `--method`: (Optional) `greedy` or `topk`. Defaults to `greedy`.

## Future Work

- **Use Relative Paths:** Update the hardcoded absolute paths in the config and scripts to be relative for better project portability.
- **Data Preparation Scripts:** Include scripts for preparing the dataset, generating CLIP embeddings, and training the BPE tokenizer.
- **Beam Search:** Implement beam search decoding in `greedy_predict.py` for potentially better caption quality.
- **More Evaluation Metrics:** Add other common captioning metrics like ROUGE, METEOR, and CIDEr using `coco_eval.py`.