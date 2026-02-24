# DeepInspire

This repository contains the code and models for the **DeepInspire** study, which focuses on automated assessment of inspiration quality in chest X-ray images using deep learning.

## Overview

DeepInspire consists of two main components:
1. **Classification Model**: Determines whether a chest X-ray shows adequate (full) or inadequate (not full) inspiration
2. **Segmentation Models**: Segments anatomical structures (lungs and ribs) from chest X-rays

## Repository Structure

```
DeepInspire/
├── train_classification.py    # Training script for classification model
├── train_segment.py           # Training script for segmentation models
├── Dataset/                   # Training and testing data
│   ├── train/                # Training images and masks
│   ├── validation/           # Validation images and masks
│   ├── test/                 # Test images and masks
│   ├── NIH/                  # NIH chest X-ray dataset
│   ├── human_segment/        # Human observer segmentations
│   └── Focus image/          # Special case images
├── Trained Model/            # Pre-trained model weights
│   ├── Classification_resnet34_20260209_051509_best.pth
│   ├── lung_MAnet_20260209_042722_best.pth
│   └── rib_MAnet_20260209_034825_best.pth
├── Notebook/                 # Jupyter notebooks for evaluation
│   ├── evaluation.ipynb      # Model evaluation and testing
│   └── visualization.ipynb   # Results visualization
└── Result/                   # Experimental results
    ├── model result/         # Model performance metrics
    ├── human result/         # Human observer performance
    ├── figure/              # Generated figures
    └── Comparison_results/  # Detailed comparison results
```

## Training

### Classification Model

Train the classification model to distinguish between adequate and inadequate inspiration:

```bash
python train_classification.py
```

**Key parameters:**
- `--train_folder`: Path to training data (default: `Dataset/train`)
- `--valid_folder`: Path to validation data (default: `Dataset/validation`)
- `--n_epochs`: Number of training epochs (default: 100)
- `--learning_rate`: Learning rate (default: 0.0001)
- `--train_batch_size`: Batch size for training (default: 4)

### Segmentation Models

Train segmentation models for lung and rib segmentation:

```bash
# Train rib segmentation model
python train_segment.py --organ rib

# Train lung segmentation model
python train_segment.py --organ lung
```

**Key parameters:**
- `--organ`: Target organ ('rib' or 'lung')
- `--train_data_dir`: Path to training data (default: `Dataset/train`)
- `--valid_data_dir`: Path to validation data (default: `Dataset/validation`)
- `--n_epochs`: Number of training epochs (default: 100)
- `--learning_rate`: Learning rate (default: 0.0001)
- `--encoder`: Encoder backbone (default: 'resnet34')

## Evaluation and Testing

For model evaluation and testing, use the Jupyter notebooks in the `Notebook/` folder:

1. **evaluation.ipynb**: Comprehensive model evaluation on test datasets
2. **visualization.ipynb**: Visualization of results and comparison with baselines

To run the notebooks:
```bash
jupyter notebook Notebook/evaluation.ipynb
```

## Dataset

The `Dataset/` folder contains:
- **train/**: Training images with full/not full inspiration labels
- **validation/**: Validation images for model tuning
- **test/**: Test images for final evaluation
- **NIH/**: Images from NIH chest X-ray dataset with various pathologies
- **human_segment/**: Manual segmentations from human observers
- **Focus image/**: Special case images including artifacts, pathologies, and challenging cases

## Results

All experimental results are stored in the `Result/` folder:
- **model result/**: Model performance metrics (CSV format)
- **human result/**: Human observer performance comparison
- **Comparison_results/**: Detailed results across different test scenarios
- **training_logs/**: Training history and logs

## Pre-trained Models

Pre-trained model weights are available in the `Trained Model/` folder:
- `Classification_resnet34_20260209_051509_best.pth`: ResNet34-based classification model
- `lung_MAnet_20260209_042722_best.pth`: MA-Net model for lung segmentation
- `rib_MAnet_20260209_034825_best.pth`: MA-Net model for rib segmentation

## Requirements

- Python 3.8+
- PyTorch
- torchvision
- segmentation-models-pytorch (smp)
- albumentations
- OpenCV
- NumPy
- scikit-learn
- matplotlib
- tqdm

## Citation

If you use this code or models in your research, please cite the DeepInspire study.

## License

This repository is provided for research purposes. Please contact the authors for licensing information.
