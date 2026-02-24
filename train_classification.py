"""
Training script for classification model (Full vs Not Full inspiration)
Based on Classification model result.ipynb
Loads images directly from folders (full/not full) like evaluation
"""

import matplotlib
matplotlib.use('Agg')  # Set non-interactive backend before importing pyplot

import torch
import glob
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import models
import os
import numpy as np
from matplotlib import pyplot as plt
import albumentations as albu
from tqdm.auto import tqdm
import time
import argparse
from pathlib import Path
import logging
from datetime import datetime
import json
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_curve, auc, precision_score, recall_score, f1_score
import seaborn as sns
import cv2


# Constants
SIZE_X = 512
SIZE_Y = 512
N_CLASSES = 2


def torch_to_np(torch_array):
    """Convert torch tensor to numpy array."""
    return np.squeeze(torch_array.detach().cpu().numpy())


def np_to_torch(np_array):
    """Convert numpy array to torch tensor."""
    return torch.from_numpy(np_array).float()


class ClassificationDataset(Dataset):
    """Dataset class for loading images from folder structure (full/not full)."""
    
    def __init__(self, image_paths, labels, augmentation=None, preprocess=None):
        """
        Args:
            image_paths: List of image file paths
            labels: Labels array (N,) with values 0 or 1
            augmentation: Albumentations augmentation pipeline
            preprocess: Preprocessing function
        """
        self.image_paths = image_paths
        self.labels = labels
        self.augmentation = augmentation
        self.preprocess = preprocess
    def __getitem__(self, index):
        # Load image from file
        img_path = self.image_paths[index]
        image = cv2.imread(img_path, 1)  # Read in BGR mode
        
        if image is None:
            raise ValueError(f"Failed to load image: {img_path}")
        
        image = cv2.resize(image, (SIZE_X, SIZE_Y), interpolation=cv2.INTER_NEAREST)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = image / 255
        image = image.astype(np.float64)  # Use float64 for higher precision
        
        label = self.labels[index]
        
        # Apply augmentations
        if self.augmentation:
            sample = self.augmentation(image=image)
            image = sample['image']
        # Apply preprocessing
        if self.preprocess:
            image = self.preprocess(image)
        
        # Convert to proper format
        image = np.transpose(image, (2, 0, 1))  # (H, W, C) -> (C, H, W)
        
        # Convert label to one-hot encoding
        label_tensor = torch.tensor(label, dtype=torch.long)
        label_onehot = F.one_hot(label_tensor, num_classes=N_CLASSES).float()
        
        image = np_to_torch(image)
        
        # Ensure contiguous for MPS compatibility
        image = image.contiguous()
        label_onehot = label_onehot.contiguous()
        
        return image, label_onehot
    
    def __len__(self):
        return len(self.image_paths)


def get_training_augmentation():
    """Get training augmentation pipeline."""
    train_transform = albu.Compose([
        albu.HorizontalFlip(),
        albu.ShiftScaleRotate(rotate_limit=5, border_mode=cv2.BORDER_CONSTANT),
        albu.RandomBrightnessContrast()
    ])
    return train_transform


def load_images_from_folders(train_folder, valid_folder):
    """
    Load image paths and labels from folder structure.
    Expected structure:
        train_folder/
            full/       (label 0 - adequate inspiration)
                image/
            not full/   (label 1 - inadequate inspiration)
                image/
        valid_folder/
            full/
                image/
            not full/
                image/
    
    Returns:
        train_paths, train_labels, valid_paths, valid_labels
    """
    print("Loading image paths from folders...")
    
    # Load training data
    train_full_folder = os.path.join(train_folder, 'full','image')
    train_notfull_folder = os.path.join(train_folder, 'not full','image')
    
    train_full_paths = sorted([os.path.join(train_full_folder, f) for f in os.listdir(train_full_folder) 
                               if f.endswith(('.png', '.jpg', '.jpeg'))])
    train_notfull_paths = sorted([os.path.join(train_notfull_folder, f) for f in os.listdir(train_notfull_folder) 
                                  if f.endswith(('.png', '.jpg', '.jpeg'))])
    
    train_paths = train_full_paths + train_notfull_paths
    train_labels = np.array([0] * len(train_full_paths) + [1] * len(train_notfull_paths), dtype=np.int64)
    
    # Shuffle training data to avoid ordered batches
    train_indices = np.random.permutation(len(train_paths))
    train_paths = [train_paths[i] for i in train_indices]
    train_labels = train_labels[train_indices]
    
    # Load validation data
    valid_full_folder = os.path.join(valid_folder, 'full','image')
    valid_notfull_folder = os.path.join(valid_folder, 'not full','image')
    
    valid_full_paths = sorted([os.path.join(valid_full_folder, f) for f in os.listdir(valid_full_folder) 
                               if f.endswith(('.png', '.jpg', '.jpeg'))])
    valid_notfull_paths = sorted([os.path.join(valid_notfull_folder, f) for f in os.listdir(valid_notfull_folder) 
                                  if f.endswith(('.png', '.jpg', '.jpeg'))])
    
    valid_paths = valid_full_paths + valid_notfull_paths
    valid_labels = np.array([0] * len(valid_full_paths) + [1] * len(valid_notfull_paths), dtype=np.int64)
    
    # Check for data leakage - verify no overlap between train and validation
    train_basenames = set([os.path.basename(p) for p in train_paths])
    valid_basenames = set([os.path.basename(p) for p in valid_paths])
    overlap = train_basenames.intersection(valid_basenames)
    if overlap:
        print(f"\n⚠️  WARNING: Found {len(overlap)} overlapping files between train and validation!")
        print(f"First few: {list(overlap)[:5]}")
    
    print(f"\nTraining data:")
    print(f"  Full (adequate): {len(train_full_paths)} images")
    print(f"  Not Full (inadequate): {len(train_notfull_paths)} images")
    print(f"  Total: {len(train_paths)} images")
    print(f"  Class balance: {len(train_full_paths)/len(train_paths):.2%} Full, {len(train_notfull_paths)/len(train_paths):.2%} Not Full")
    
    print(f"\nValidation data:")
    print(f"  Full (adequate): {len(valid_full_paths)} images")
    print(f"  Not Full (inadequate): {len(valid_notfull_paths)} images")
    print(f"  Total: {len(valid_paths)} images")
    print(f"  Class balance: {len(valid_full_paths)/len(valid_paths):.2%} Full, {len(valid_notfull_paths)/len(valid_paths):.2%} Not Full\n")
    
    # Sample a few images to verify they load correctly
    print("Verifying data loading...")
    for i in [0, len(train_paths)//2, len(train_paths)-1]:
        img = cv2.imread(train_paths[i])
        if img is None:
            print(f"  ❌ ERROR: Cannot load {train_paths[i]}")
        else:
            print(f"  ✓ Sample train image {i}: shape={img.shape}, label={train_labels[i]}")
    
    for i in [0, len(valid_paths)-1]:
        img = cv2.imread(valid_paths[i])
        if img is None:
            print(f"  ❌ ERROR: Cannot load {valid_paths[i]}")
        else:
            print(f"  ✓ Sample valid image {i}: shape={img.shape}, label={valid_labels[i]}")
    print()
    
    return train_paths, train_labels, valid_paths, valid_labels


def calculate_metrics(true_labels, pred_labels, pred_probs):
    """
    Calculate classification metrics.
    
    Args:
        true_labels: True class labels
        pred_labels: Predicted class labels
        pred_probs: Predicted probabilities for positive class
    
    Returns:
        Dictionary of metrics
    """
    cm = confusion_matrix(true_labels, pred_labels)
    tn, fp, fn, tp = cm.ravel()
    
    accuracy = accuracy_score(true_labels, pred_labels)
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0  # Recall, TPR - detects Inadequate (Positive)
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0  # TNR - detects Adequate (Negative)
    precision = precision_score(true_labels, pred_labels, zero_division=0)
    f1 = f1_score(true_labels, pred_labels, zero_division=0)
        
    # Calculate AUC
    fpr, tpr, _ = roc_curve(true_labels, pred_probs)
    roc_auc = auc(fpr, tpr)
    
    metrics = {
        'accuracy': accuracy,
        'sensitivity': sensitivity,
        'specificity': specificity,
        'precision': precision,
        'f1_score': f1,
        'auc': roc_auc,
        'confusion_matrix': cm.tolist(),
        'tp': int(tp),
        'tn': int(tn),
        'fp': int(fp),
        'fn': int(fn)
    }
    
    return metrics


def setup_logger(log_dir, model_name):
    """Setup logger for training."""
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    log_file = os.path.join(log_dir, f'{model_name}_training.log')
    
    # Create logger
    logger = logging.getLogger(model_name)
    logger.setLevel(logging.INFO)
    
    # Remove existing handlers
    logger.handlers = []
    
    # File handler
    fh = logging.FileHandler(log_file)
    fh.setLevel(logging.INFO)
    
    # Console handler
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    
    # Formatter
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    
    logger.addHandler(fh)
    logger.addHandler(ch)
    
    return logger, log_file


def train_model(model, train_loader, valid_loader, criterion, optimizer, scheduler, device, 
                n_epochs=100, display_step=5, save_dir='trained_models', 
                model_name='classification_model', save_fig=False, fig_dir='train_history',
                log_dir='training_logs'):
    """
    Train classification model with validation and save best model.
    
    Args:
        model: PyTorch model
        train_loader: Training data loader
        valid_loader: Validation data loader
        criterion: Loss function
        optimizer: Optimizer
        device: Device to train on
        n_epochs: Number of epochs
        display_step: Display frequency
        save_dir: Directory to save models
        model_name: Name for saved model
        save_fig: Whether to save training figures
        fig_dir: Directory to save figures
        log_dir: Directory to save training logs
    """
    # Create directories
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    if save_fig:
        Path(fig_dir).mkdir(parents=True, exist_ok=True)
    
    # Setup logger
    logger, log_file = setup_logger(log_dir, model_name)
    
    plt_loss = []
    plt_val_loss = []
    accuracy_history = []
    epoch_times = []
    mean_loss = 0
    best_val_loss = float('inf')
    best_accuracy = 0.0
    best_epoch = 0
    
    # Log training start
    logger.info("="*70)
    logger.info(f"Training {model_name}")
    logger.info("="*70)
    logger.info(f"Device: {device}")
    logger.info(f"Epochs: {n_epochs}")
    logger.info(f"Train batches: {len(train_loader)}")
    logger.info(f"Valid batches: {len(valid_loader)}")
    logger.info(f"Train dataset size: {len(train_loader.dataset)}")
    logger.info(f"Valid dataset size: {len(valid_loader.dataset)}")
    logger.info(f"Batch size: {train_loader.batch_size}")
    logger.info(f"Optimizer: {type(optimizer).__name__}")
    logger.info(f"Loss function: BCEWithLogitsLoss")
    logger.info(f"Architecture: {type(model).__name__}")
    logger.info(f"Scheduler: {type(scheduler).__name__ if scheduler is not None else 'None'}")
    logger.info("="*70)
    
    # Start training timer
    training_start_time = time.time()
    
    for epoch in range(n_epochs + 1):
        epoch_start_time = time.time()
        
        # Training phase
        model.train()
        running_loss = 0.0
        
        for x, y in tqdm(train_loader, desc=f'Epoch {epoch}/{n_epochs} [Train]'):
            y = y.to(device)
            x = x.to(device)
            
            # Update model
            model.zero_grad()
            pred = model(x)
            loss = criterion(pred, y)
            loss.backward()
            optimizer.step()
            
            # Track loss
            mean_loss += loss.item() / round(len(train_loader.dataset) / train_loader.batch_size)
            running_loss += loss.item() * x.size(0)
        
        epoch_train_loss = running_loss / len(train_loader.dataset)
        plt_loss.append(epoch_train_loss)
        
        # Validation phase
        model.eval()
        running_loss = 0.0
        all_true_labels = []
        all_pred_labels = []
        all_pred_probs = []
        
        # Verify model is in eval mode (BatchNorm, Dropout behave differently)
        if epoch == 0:
            is_training = model.training
            logger.info(f"  Model training mode during validation: {is_training} (should be False)")
        
        with torch.no_grad():
            for xv, yv in tqdm(valid_loader, desc=f'Epoch {epoch}/{n_epochs} [Valid]'):
                yv = yv.to(device)
                xv = xv.to(device)
                predv = model(xv)
                val_loss = criterion(predv, yv)
                running_loss += val_loss.item() * xv.size(0)
                
                # Get predictions and probabilities
                pred_probs = torch.softmax(predv, dim=1).cpu().numpy()
                pred_labels = torch.argmax(predv, dim=1).cpu().numpy()
                true_labels = np.argmax(yv.cpu().numpy(), axis=1)
                
                all_pred_probs.extend(pred_probs[:, 1])  # Probability for class 1 (not full)
                all_pred_labels.extend(pred_labels)
                all_true_labels.extend(true_labels)
        
        epoch_valid_loss = running_loss / len(valid_loader.dataset)
        plt_val_loss.append(epoch_valid_loss)
        
        # Calculate metrics
        metrics = calculate_metrics(all_true_labels, all_pred_labels, all_pred_probs)
        accuracy_history.append(metrics['accuracy'])
        
        # Calculate epoch time
        epoch_time = time.time() - epoch_start_time
        epoch_times.append(epoch_time)
        
        # Get current learning rate
        current_lr = optimizer.param_groups[0]['lr']
        
        # Log epoch results with LR
        logger.info(f"Epoch {epoch}/{n_epochs} - "
                   f"Train Loss: {epoch_train_loss:.5f} | "
                   f"Val Loss: {epoch_valid_loss:.5f} | "
                   f"Acc: {metrics['accuracy']:.4f} | "
                   f"F1: {metrics['f1_score']:.4f} | "
                   f"AUC: {metrics['auc']:.4f} | "
                   f"LR: {current_lr:.2e} | "
                   f"Time: {epoch_time:.2f}s")
        
        # Add diagnostic information on first epoch
        if epoch == 0:
            logger.info(f"  → First epoch diagnostics:")
            logger.info(f"     Train samples: {len(train_loader.dataset)}")
            logger.info(f"     Valid samples: {len(valid_loader.dataset)}")
            logger.info(f"     Train/Val loss ratio: {epoch_train_loss/epoch_valid_loss:.3f}")
            logger.info(f"     Confusion Matrix: TN={metrics['tn']}, FP={metrics['fp']}, FN={metrics['fn']}, TP={metrics['tp']}")
            if epoch_valid_loss > epoch_train_loss * 1.5:
                logger.warning(f"  ⚠️  Validation loss is significantly higher than training loss!")
                logger.warning(f"     This could indicate: data mismatch, different distributions, or validation is harder")
        
        # Save best model based on validation loss
        if epoch_valid_loss < best_val_loss:
            best_val_loss = epoch_valid_loss
            best_accuracy = metrics['accuracy']
            best_epoch = epoch
            best_model_path = os.path.join(save_dir, f'{model_name}_best.pth')
            torch.save({
                'epoch': epoch,
                'model': model.state_dict(),
                'opt': optimizer.state_dict(),
                'train_loss': epoch_train_loss,
                'val_loss': epoch_valid_loss,
                'metrics': metrics
            }, best_model_path)
            logger.info(f"✓ Best model saved at epoch {epoch} with val_loss: {epoch_valid_loss:.5f}, accuracy: {metrics['accuracy']:.4f}")
        
        # Update learning rate scheduler
        if scheduler is not None:
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(epoch_valid_loss)
            else:
                scheduler.step()
        
        # Check for overfitting after a few epochs
        if epoch >= 5:
            # If train loss is decreasing but val loss is increasing/stagnant
            if len(plt_loss) >= 5 and len(plt_val_loss) >= 5:
                recent_train_trend = plt_loss[-1] - plt_loss[-5]  # Should be negative (decreasing)
                recent_val_trend = plt_val_loss[-1] - plt_val_loss[-5]  # Should be negative (decreasing)
                
                if recent_train_trend < -0.01 and recent_val_trend > 0.01:
                    logger.warning(f"  ⚠️  Possible overfitting detected at epoch {epoch}:")
                    logger.warning(f"     Train loss decreased by {abs(recent_train_trend):.4f} but val loss increased by {recent_val_trend:.4f}")
                    logger.warning(f"     Consider: reducing model complexity, adding regularization, or early stopping")
        
        mean_loss = 0
    
    # Calculate total training time
    total_training_time = time.time() - training_start_time
    avg_epoch_time = np.mean(epoch_times)
    
    # Final evaluation on validation set
    print("\nGenerating final evaluation...")
    model.eval()
    all_true_labels = []
    all_pred_labels = []
    all_pred_probs = []
    
    with torch.no_grad():
        for x, y in valid_loader:
            x = x.to(device)
            y = y.to(device)
            pred = model(x)
            
            pred_probs = torch.softmax(pred, dim=1).cpu().numpy()
            pred_labels = np.argmax(pred_probs, axis=1)
            true_labels = np.argmax(y.cpu().numpy(), axis=1)
            
            all_pred_probs.extend(pred_probs[:, 1])
            all_pred_labels.extend(pred_labels)
            all_true_labels.extend(true_labels)
    
    final_metrics = calculate_metrics(all_true_labels, all_pred_labels, all_pred_probs)
    
    # Create final visualization
    fig = plt.figure(figsize=(20, 10))
    
    # Plot 1: Training and Validation Loss
    plt.subplot(2, 4, 1)
    plt.plot(plt_loss, 'r-', linewidth=2, label='Train Loss')
    plt.plot(plt_val_loss, 'b-', linewidth=2, label='Val Loss')
    plt.axvline(x=best_epoch, color='g', linestyle='--', linewidth=2, label=f'Best Epoch ({best_epoch})')
    plt.legend()
    plt.title('Loss Curves', fontweight='bold')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True, alpha=0.3)
    
    # Plot 2: Accuracy over epochs
    plt.subplot(2, 4, 2)
    plt.plot(accuracy_history, linewidth=2, color='green')
    plt.axvline(x=best_epoch, color='g', linestyle='--', linewidth=1, alpha=0.5)
    plt.title('Validation Accuracy', fontweight='bold')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.grid(True, alpha=0.3)
    plt.ylim([0, 1])
    
    # Plot 3: Confusion Matrix
    plt.subplot(2, 4, 3)
    cm = np.array(final_metrics['confusion_matrix'])
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False, square=True,
                xticklabels=['Full', 'Not Full'], yticklabels=['Full', 'Not Full'])
    plt.xlabel('Predicted labels')
    plt.ylabel('True labels')
    plt.title('Confusion Matrix', fontweight='bold')
    
    # Plot 4: ROC Curve
    plt.subplot(2, 4, 4)
    fpr, tpr, _ = roc_curve(all_true_labels, all_pred_probs)
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve', fontweight='bold')
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    
    # Plot 5-8: Sample predictions
    model.eval()
    with torch.no_grad():
        for x, y in valid_loader:
            x = x.to(device)
            y = y.to(device)
            pred = model(x)
            break
    
    class_names = ['Full', 'Not Full']
    for i in range(min(4, len(x))):
        plt.subplot(2, 4, 5 + i)
        img_np = torch_to_np(x[i][0])
        true_label = np.argmax(y[i].cpu().numpy())
        pred_probs = torch.softmax(pred[i], dim=0).cpu().numpy()
        pred_label = np.argmax(pred_probs)
        
        plt.imshow(img_np, cmap='gray')
        color = 'green' if pred_label == true_label else 'red'
        plt.title(f'True: {class_names[true_label]}\\nPred: {class_names[pred_label]} ({pred_probs[pred_label]:.2f})',
                 color=color, fontweight='bold')
        plt.axis('off')
    
    plt.suptitle(f'Training Results - {model_name}', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    # Save final visualization
    final_viz_path = os.path.join(log_dir, f'{model_name}_final_visualization.png')
    plt.savefig(final_viz_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Final visualization saved to: {final_viz_path}")
    
    # Save final model
    final_model_path = os.path.join(save_dir, f'{model_name}_final.pth')
    torch.save({
        'epoch': n_epochs,
        'model': model.state_dict(),
        'opt': optimizer.state_dict(),
        'train_loss': epoch_train_loss,
        'val_loss': epoch_valid_loss,
        'metrics': final_metrics
    }, final_model_path)
    
    # Save training history
    history = {
        'train_loss': [float(loss) for loss in plt_loss],
        'val_loss': [float(loss) for loss in plt_val_loss],
        'accuracy': [float(acc) for acc in accuracy_history],
        'epoch_times': [float(t) for t in epoch_times],
        'best_epoch': int(best_epoch),
        'best_val_loss': float(best_val_loss),
        'best_accuracy': float(best_accuracy),
        'final_metrics': {k: float(v) if isinstance(v, (int, float, np.number)) else v 
                         for k, v in final_metrics.items()},
        'total_training_time': float(total_training_time),
        'avg_epoch_time': float(avg_epoch_time)
    }
    history_path = os.path.join(log_dir, f'{model_name}_history.json')
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=4)
    
    # Log training summary
    logger.info("="*70)
    logger.info("TRAINING COMPLETE")
    logger.info("="*70)
    logger.info(f"Total training time: {time.strftime('%H:%M:%S', time.gmtime(total_training_time))}")
    logger.info(f"Average epoch time: {avg_epoch_time:.2f}s")
    logger.info(f"Best epoch: {best_epoch}")
    logger.info(f"Best validation loss: {best_val_loss:.5f}")
    logger.info(f"Best accuracy: {best_accuracy:.4f}")
    logger.info(f"Final metrics:")
    logger.info(f"  Accuracy: {final_metrics['accuracy']:.4f}")
    logger.info(f"  Sensitivity: {final_metrics['sensitivity']:.4f}")
    logger.info(f"  Specificity: {final_metrics['specificity']:.4f}")
    logger.info(f"  Precision: {final_metrics['precision']:.4f}")
    logger.info(f"  F1-Score: {final_metrics['f1_score']:.4f}")
    logger.info(f"  AUC: {final_metrics['auc']:.4f}")
    logger.info(f"Final model saved to: {final_model_path}")
    logger.info(f"Best model saved to: {best_model_path}")
    logger.info(f"Training history saved to: {history_path}")
    logger.info(f"Training log saved to: {log_file}")
    logger.info("="*70)
    
    return plt_loss, plt_val_loss, best_epoch, best_val_loss, final_metrics


def train_classification_model(train_folder='Dataset/Classification model/train image',
                               valid_folder='Dataset/Classification model/validation image',
                               train_batch_size=4, n_epochs=100, display_step=5,
                               learning_rate=0.0001, save_dir='trained_models',
                               save_fig=False, fig_dir='train_history',
                               log_dir='training_logs'):
    """
    Main function to train classification model.
    
    Args:
        train_folder: Directory containing train images with 'full' and 'not full' subfolders
        valid_folder: Directory containing validation images with 'full' and 'not full' subfolders
        train_batch_size: Batch size for training
        n_epochs: Number of training epochs
        display_step: Display frequency
        learning_rate: Learning rate
        save_dir: Directory to save trained models
        save_fig: Whether to save training figures
        fig_dir: Directory to save figures
        log_dir: Directory to save training logs
    """
    print(f"\n{'='*70}")
    print(f"TRAINING CLASSIFICATION MODEL (Full vs Not Full)")
    print(f"{'='*70}\n")
    
    # Load image paths from folders (same as evaluation approach)
    train_paths, train_labels, valid_paths, valid_labels = load_images_from_folders(train_folder, valid_folder)
    
    # Create datasets
    train_dataset = ClassificationDataset(
        train_paths, train_labels,
        augmentation=get_training_augmentation()
    )
    valid_dataset = ClassificationDataset(
        valid_paths, valid_labels,
        augmentation=None  # No augmentation for validation to match evaluation
    )
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=1, shuffle=False)
    
    # Setup device (with MPS support for Mac)
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    
    print(f"Using device: {device}\n")
    
    # Create model
    print("Creating ResNet34 model...")
    model = models.resnet34(weights='IMAGENET1K_V1')
    
    # Modify the final layer for binary classification (add dropout)
    model.fc = nn.Sequential(
        nn.Dropout(p=0.4),
        nn.Linear(model.fc.in_features, N_CLASSES)
    )
    model = model.to(device)
    
    # Setup loss and optimizer
    criterion = nn.BCEWithLogitsLoss()
    print("----------------------------------")
    print(f"Using loss function: BCEWithLogitsLoss\n")
    print("----------------------------------\n")
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, betas=(0.5, 0.999))
    
    # Setup learning rate scheduler
    # Option 1: ReduceLROnPlateau - reduces LR when validation loss plateaus (RECOMMENDED)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, verbose=True, min_lr=1e-7
    )
    
    # Option 2: CosineAnnealingLR - smooth cosine decay
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    #     optimizer, T_max=n_epochs, eta_min=1e-7
    # )
    
    # Option 3: CosineAnnealingWarmRestarts - periodic restarts (ACTIVE)
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
    #     optimizer, T_0=10, T_mult=2, eta_min=1e-7
    # )
    
    # Option 4: StepLR - step decay every N epochs
    # scheduler = torch.optim.lr_scheduler.StepLR(
    #     optimizer, step_size=30, gamma=0.5
    # )
    
    # Option 5: OneCycleLR - one cycle policy (good for fast training)
    # scheduler = torch.optim.lr_scheduler.OneCycleLR(
    #     optimizer, max_lr=learning_rate*10, 
    #     steps_per_epoch=len(train_loader), epochs=n_epochs
    # )
    
    # Option 6: No scheduler
    # scheduler = None
    print("----------------------------------")
    print(f"Using scheduler: {type(scheduler).__name__ if scheduler else 'None'}\n")
    print("----------------------------------\n")
    # Model name
    model_name = f'Classification_resnet34_{datetime.now().strftime("%Y%m%d_%H%M%S")}'
    
    # Train model
    train_loss, val_loss, best_epoch, best_val_loss, final_metrics = train_model(
        model=model,
        train_loader=train_loader,
        valid_loader=valid_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        n_epochs=n_epochs,
        display_step=display_step,
        save_dir=save_dir,
        model_name=model_name,
        save_fig=save_fig,
        fig_dir=fig_dir,
        log_dir=log_dir
    )
    
    return model, train_loss, val_loss, final_metrics


def main():
    """Main function for command-line usage."""
    parser = argparse.ArgumentParser(description='Train classification model')
    parser.add_argument('--train_folder', type=str, default='../2026 Dataset/train',
                       help='Directory containing training images with full/not full subfolders')
    parser.add_argument('--valid_folder', type=str, default='../2026 Dataset/validation',
                       help='Directory containing validation images with full/not full subfolders')
    parser.add_argument('--batch_size', type=int, default=4,
                       help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=100,
                       help='Number of training epochs')
    parser.add_argument('--display_step', type=int, default=5,
                       help='Display frequency')
    parser.add_argument('--lr', type=float, default=0.0001,
                       help='Learning rate')
    parser.add_argument('--save_dir', type=str, default='../train_result/trained_models',
                       help='Directory to save trained models')
    parser.add_argument('--save_fig', action='store_true',
                       help='Save training figures')
    parser.add_argument('--fig_dir', type=str, default='../train_result/train_history',
                       help='Directory to save training figures')
    parser.add_argument('--log_dir', type=str, default='../train_result/training_logs',
                       help='Directory to save training logs')
    
    args = parser.parse_args()
    
    train_classification_model(
        train_folder=args.train_folder,
        valid_folder=args.valid_folder,
        train_batch_size=args.batch_size,
        n_epochs=args.epochs,
        display_step=args.display_step,
        learning_rate=args.lr,
        save_dir=args.save_dir,
        save_fig=args.save_fig,
        fig_dir=args.fig_dir,
        log_dir=args.log_dir
    )


if __name__ == '__main__':
    print("="*70)
    print("CLASSIFICATION MODEL TRAINING SCRIPT")
    print("="*70)
    print("\nUsage examples:")
    print("\n1. Train with default settings:")
    print("   python train_classification.py")
    print("\n2. Custom training:")
    print("   python train_classification.py --epochs 50 --batch_size 8 --lr 0.001")
    print("\n3. With figure saving:")
    print("   python train_classification.py --save_fig")
    print("\n" + "="*70 + "\n")
    
    main()
