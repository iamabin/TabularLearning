# Tabular Deep Learning: Unsupervised Pre-training on Structured Data

This project explores the feasibility and effectiveness of **unsupervised pre-training** for **tabular data**, a common data type in real-world applications. Inspired by the success of self-supervised learning in vision and language, the project investigates whether similar techniques can be applied to structured, mixed-type tabular datasets using a Transformer-based architecture.

### Pre-training Tasks

- Masked Feature Modeling (MFM):
  - Inspired by BERT's Masked Language Model (MLM)
  - Features are randomly masked and the model is trained to reconstruct them
- Two Masking Strategies:
  - Unified token masking: a shared learnable embedding for all masked features
  - Multi-token masking: position-specific masking embeddings for more precision

### Architecture

- Encoder: Multi-layer Transformer (attention + feedforward)
- Embedding layers: Separate modules for categorical and continuous inputs
- Prediction heads:
  - MLP per feature (regression for continuous, classification for categorical)
- Losses: MSE for continuous, CrossEntropy for categorical features

### Set Up

**Dependencies**

```
pip install torch pandas scikit-learn
pip install torch torchvision --extra-index-url https://download.pytorch.org/whl/cu118
```

**Run training**

```
python train.py
```

You can manually configure the following parameters inside `train.py`:

- Dataset path (e.g., `data/1995_income.csv`)
- Batch size / Learning rate
- Model type (e.g., with or without pretraining)
- Masking ratio / Random seed, etc.

### Report

Full project report included in `TabularDeepLearning.pdf`, containing methodology, experiments, results, and figures.
