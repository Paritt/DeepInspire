"""
Training script for segmentation models (Rib and Lung)
Based on 2024train rib lung model newdataset.ipynb
Loads .png images and .tiff masks directly from folder structure
"""

import matplotlib

matplotlib.use('Agg')  # Set non-interactive backend before importing pyplot

import torch
import segmentation_models_pytorch as smp
import os
import numpy as np
from matplotlib import pyplot as plt
import albumentations as albu
from tqdm.auto import tqdm
from torch.utils.data import DataLoader, Dataset
import time
import torch.nn as nn
import argparse
from pathlib import Path
import logging
from datetime import datetime
import json
import cv2
from scipy.ndimage import distance_transform_edt
import glob
import tifffile as tiff
from typing import List, Sequence
from scipy.ndimage import label as cc_label


# Constants
SIZE_X = 512
SIZE_Y = 512
N_CLASSES = 3


def torch_to_np(torch_array):
    """Convert torch tensor to numpy array."""
    return np.squeeze(torch_array.detach().cpu().numpy())


def np_to_torch(np_array):
    """Convert numpy array to torch tensor."""
    return torch.from_numpy(np_array).float()


def load_image(images_path):
    """Load images from folder.
    
    Args:
        images_path: Path pattern to image folder (e.g., 'train/full/image')
    
    Returns:
        train_images: numpy array of images (N, H, W, C)
        train_images_path: list of image paths
    """
    train_images_path = []
    train_images = []
    
    for directory_path in glob.glob(images_path):
        for img_path in glob.glob(os.path.join(directory_path, "*.png")):
            train_images_path.append(img_path)
    
    train_images_path.sort()
    
    for img_path in train_images_path:
        img = cv2.imread(img_path, 1)
        img = cv2.resize(img, (SIZE_X, SIZE_Y), interpolation=cv2.INTER_NEAREST)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img / 255.0
        train_images.append(img)
    
    train_images = np.array(train_images)
    return train_images, train_images_path


def load_mask(masks_path):
    """Load masks from folder.
    
    Args:
        masks_path: Path pattern to mask folder (e.g., 'train/full/mask/rib')
    
    Returns:
        train_masks: numpy array of masks (N, H, W, C)
        train_masks_path: list of mask paths
    """
    train_masks_path = []
    train_masks = []
    
    for directory_path in glob.glob(masks_path):
        for mask_path in glob.glob(os.path.join(directory_path, "*.tiff")):
            train_masks_path.append(mask_path)
    
    train_masks_path.sort()
    
    for mask_path in train_masks_path:
        mask = np.array(tiff.imread(mask_path))
        mask = cv2.resize(mask, (SIZE_X, SIZE_Y), interpolation=cv2.INTER_NEAREST)
        train_masks.append(mask)
    
    train_masks = np.array(train_masks)
    return train_masks, train_masks_path


def dice_cal(y_true, y_pred):
    """Calculate Dice Similarity Coefficient between two binary masks."""
    intersection = np.sum(np.logical_and(y_true, y_pred).astype(float))
    if (np.sum(y_true) == 0) and (np.sum(y_pred) == 0):
        return 1.0
    if (np.sum(y_true) + np.sum(y_pred)) == 0:
        return 0.0
    return (2 * intersection) / (np.sum(y_true) + np.sum(y_pred))

def calculate_dsc_per_class(pred_argmax, target, num_classes=3):
    """Calculate DSC for each class.
    
    Args:
        pred_argmax: Predicted mask argmax (H, W)
        target: Ground truth mask (H, W, C)
        num_classes: Number of classes
        threshold: Threshold for binary prediction
    
    Returns:
        List of DSC values for each class
    """
    dsc_per_class = []
    
    for class_idx in range(num_classes):
        dsc = dice_cal(target[class_idx,:,:], pred_argmax == class_idx)
        
        dsc_per_class.append(dsc)
    
    return dsc_per_class


class PreprocessedDataset(Dataset):
    """Dataset class for loading preprocessed .npy data."""
    
    def __init__(self, images, masks, augmentation=None, preprocess=None):
        """
        Args:
            images: Preprocessed images array (N, H, W, C)
            masks: Masks array (N, H, W, C)
            augmentation: Albumentations augmentation pipeline
            preprocess: Segmentation preprocessing
        """
        self.images = images
        self.masks = masks
        self.augmentation = augmentation
        self.preprocess = preprocess

    def __getitem__(self, index):
        image = self.images[index]
        mask = self.masks[index]
        
        # Apply augmentations
        if self.augmentation:
            sample = self.augmentation(image=image.astype(np.float32), 
                                      mask=mask.astype(np.uint8))
            image, mask = sample['image'], sample['mask']
        # Apply preprocessing
        if self.preprocess:
            image = self.preprocess(image)
        
        # Convert to proper format
        image = image.astype(np.float64)
        image = np.transpose(image, (2, 0, 1))  # (H, W, C) -> (C, H, W)
        mask = np.transpose(mask, (2, 0, 1))    # (H, W, C) -> (C, H, W)
        
        image = np_to_torch(image)
        mask = np_to_torch(mask)
        
        return image, mask
    
    def __len__(self):
        return min(len(self.images), len(self.masks))


def get_training_augmentation():
    """Get training augmentation pipeline."""
    train_transform = albu.Compose([
        albu.HorizontalFlip(),
        albu.ShiftScaleRotate(rotate_limit=5, border_mode=cv2.BORDER_CONSTANT),
        albu.RandomBrightnessContrast()
    ], additional_targets={'mask': 'mask'})
    return albu.Compose(train_transform)

# --------------
# Loss Functions
# --------------

def sub_dice_loss(y_pred, y_true):
    """Dice loss function that only uses channels 1 and 2 (excludes background)."""
    y_pred_subset = y_pred[:, 1:3, :, :].contiguous()
    y_true_subset = y_true[:, 1:3, :, :].contiguous()
    dice_loss = smp.losses.DiceLoss(mode='multilabel')
    loss = dice_loss(y_pred_subset, y_true_subset)
    return loss

def weighted_dice_loss(y_pred, y_true, class_weights=(0.01, 1, 2)):
    """Weighted Dice loss with manual per-channel class weights (bg, left, right).

    This implementation loops over channels and applies the weights because
    the installed SMP DiceLoss version does not support `class_weights`.
    """
    weights = torch.tensor(class_weights, device=y_pred.device, dtype=torch.float32)

    base_dice = smp.losses.DiceLoss(mode='multilabel', from_logits=True)

    total_loss = 0.0
    for c in range(y_pred.shape[1]):  # iterate over 3 channels
        y_pred_c = y_pred[:, c:c+1, :, :].contiguous()
        y_true_c = y_true[:, c:c+1, :, :].contiguous()

        dice_l = base_dice(y_pred_c, y_true_c)

        total_loss += weights[c] * dice_l

    def __name__(self):
        return "weighted_dice_loss"
    
    return total_loss / weights.sum()


def combined_bce_dice_loss(y_pred, y_true):
    """Combined BCE + Dice loss for channels 1 and 2 (excludes background)."""
    y_pred_subset = y_pred[:, 1:3, :, :].contiguous()
    y_true_subset = y_true[:, 1:3, :, :].contiguous()
    
    dice_loss = smp.losses.DiceLoss(mode='multilabel')
    bce_loss = smp.losses.SoftBCEWithLogitsLoss()
    
    loss = 0.5 * dice_loss(y_pred_subset, y_true_subset) + 0.5 * bce_loss(y_pred_subset, y_true_subset)
    return loss


def focal_dice_loss(y_pred, y_true, class_weights=(0.01, 1, 2)):
    """Focal + Dice loss with manual per-channel class weights (bg, left, right).

    This implementation loops over channels and applies the weights because
    the installed SMP DiceLoss/FocalLoss versions do not support `class_weights`.
    """
    weights = torch.tensor(class_weights, device=y_pred.device, dtype=torch.float32)

    base_dice = smp.losses.DiceLoss(mode='multilabel', from_logits=True)
    base_focal = smp.losses.FocalLoss(mode='multilabel')

    total_loss = 0.0
    for c in range(y_pred.shape[1]):  # iterate over 3 channels
        y_pred_c = y_pred[:, c:c+1, :, :].contiguous()
        y_true_c = y_true[:, c:c+1, :, :].contiguous()

        dice_l = base_dice(y_pred_c, y_true_c)
        focal_l = base_focal(y_pred_c, y_true_c)

        total_loss += weights[c] * (0.5 * dice_l + 0.5 * focal_l)

    def __name__(self):
        return "focal_dice_loss"
    
    return total_loss / weights.sum()

dice_loss = smp.losses.DiceLoss('multilabel', from_logits=True)
bce_loss = smp.losses.SoftBCEWithLogitsLoss()

class InverseDiceLoss(nn.Module):
    def __init__(self):
        super(InverseDiceLoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        inputs = torch.sigmoid(inputs)
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        intersection = (inputs * targets).sum()
        dice = (2. * intersection + smooth) / (inputs.sum() + targets.sum() + smooth)
        loss = 1 - dice
        return loss

inverse_dice_loss = InverseDiceLoss()

logits = torch.randn(4, 3, 512, 512)  
targets = torch.randint(0, 2, (4, 3, 512, 512), dtype=torch.float32) 

def dice_bce_inverse_loss_2channel(y_pred, y_true):
    y_pred_subset = y_pred[:, 1:3, :, :].contiguous()
    y_true_subset = y_true[:, 1:3, :, :].contiguous()
    loss1 = dice_loss(y_pred_subset, y_true_subset)
    loss2 = bce_loss(y_pred_subset, y_true_subset)
    loss3 = inverse_dice_loss(y_pred_subset, y_true_subset)
    
    total_loss = (1/3) * loss1 + (1/3) * loss2 + (1/3) * loss3
    return total_loss

def dice_bce_inverse_loss_3channel(y_pred, y_true):
    loss1 = dice_loss(y_pred, y_true)
    loss2 = bce_loss(y_pred, y_true)
    loss3 = inverse_dice_loss(y_pred, y_true)
    
    total_loss = (1/3) * loss1 + (1/3) * loss2 + (1/3) * loss3
    return total_loss


class BoundaryLoss(nn.Module):
    """Boundary Loss for segmentation - emphasizes accurate boundaries."""
    
    def __init__(self):
        super(BoundaryLoss, self).__init__()
    
    def compute_distance_map(self, mask):
        """Compute distance transform for each channel.
        Args:
            mask: (B, C, H, W) binary mask
        Returns:
            distance_map: (B, C, H, W) distance from boundaries
        """
        batch_size, num_classes, height, width = mask.shape
        distance_maps = torch.zeros_like(mask)
        
        for b in range(batch_size):
            for c in range(num_classes):
                # Convert to numpy for distance transform
                mask_np = mask[b, c].cpu().numpy()
                
                # Compute distance transform
                # Distance from foreground pixels to background
                if mask_np.max() > 0:
                    # Distance inside the object (positive)
                    dist_inside = distance_transform_edt(mask_np)
                    # Distance outside the object (negative)
                    dist_outside = distance_transform_edt(1 - mask_np)
                    # Signed distance map (positive inside, negative outside)
                    distance_map = dist_inside - dist_outside
                else:
                    distance_map = -distance_transform_edt(1 - mask_np)
                
                distance_maps[b, c] = torch.from_numpy(distance_map).float()
        
        return distance_maps.to(mask.device)
    
    def forward(self, pred, target):
        """
        Args:
            pred: (B, C, H, W) predicted probabilities (after sigmoid)
            target: (B, C, H, W) ground truth binary masks
        """
        # Apply sigmoid to predictions if not already
        pred = torch.sigmoid(pred)
        
        # Compute distance map from target boundaries
        with torch.no_grad():
            distance_map = self.compute_distance_map(target)
        
        # Boundary loss: weighted by distance to boundary
        # Areas close to boundary have higher weights
        boundary_weights = torch.abs(distance_map)
        boundary_weights = torch.exp(-boundary_weights / 10.0)  # Normalize and emphasize boundaries
        
        # Compute weighted BCE
        bce = -(target * torch.log(pred + 1e-7) + (1 - target) * torch.log(1 - pred + 1e-7))
        weighted_bce = bce * boundary_weights
        
        loss = weighted_bce.mean()
        return loss


def boundary_dice_loss(y_pred, y_true):
    """Boundary + Dice loss for channels 1 and 2 - STATE-OF-THE-ART for precise boundaries."""
    y_pred_subset = y_pred[:, 1:3, :, :].contiguous()
    y_true_subset = y_true[:, 1:3, :, :].contiguous()
    
    dice_loss = smp.losses.DiceLoss(mode='multilabel')
    boundary_loss = BoundaryLoss()
    
    loss_dice = dice_loss(y_pred_subset, y_true_subset)
    loss_boundary = boundary_loss(y_pred_subset, y_true_subset)
    
    # Weighted combination: 60% Dice + 40% Boundary
    total_loss = 0.6 * loss_dice + 0.4 * loss_boundary
    return total_loss


def boundary_focal_dice_loss(y_pred, y_true):
    """Boundary + Focal + Dice loss - ULTIMATE combination for medical segmentation."""
    y_pred_subset = y_pred[:, 1:3, :, :].contiguous()
    y_true_subset = y_true[:, 1:3, :, :].contiguous()
    
    dice_loss = smp.losses.DiceLoss(mode='multilabel')
    focal_loss = smp.losses.FocalLoss(mode='multilabel')
    boundary_loss = BoundaryLoss()
    
    loss_dice = dice_loss(y_pred_subset, y_true_subset)
    loss_focal = focal_loss(y_pred_subset, y_true_subset)
    loss_boundary = boundary_loss(y_pred_subset, y_true_subset)
    
    # Weighted combination: 20% Dice + 20% Focal + 60% Boundary
    total_loss = 0.2 * loss_dice + 0.2 * loss_focal + 0.6 * loss_boundary
    return total_loss


def boundary_focal_dice_cc_loss(y_pred, y_true, class_weights= (0.00001, 1.0, 1.0)):
    """Boundary + Focal + Dice + CC loss for medical segmentation (3-channel)."""
    y_pred_subset = y_pred[:, :, :, :].contiguous()
    y_true_subset = y_true[:, :, :, :].contiguous()
    
    dice_loss = smp.losses.DiceLoss(mode='multilabel')
    focal_loss = smp.losses.FocalLoss(mode='multilabel')
    boundary_loss = BoundaryLoss()
    
    loss_dice = dice_loss(y_pred_subset, y_true_subset)
    loss_focal = focal_loss(y_pred_subset, y_true_subset)
    loss_boundary = boundary_loss(y_pred_subset, y_true_subset)

    # Connected component loss (per-class, weighted) using argmax
    weights = torch.tensor(class_weights, device=y_pred.device, dtype=torch.float32)
    probs = torch.softmax(y_pred_subset, dim=1)
    pred_label = probs.argmax(dim=1)
    pred_one_hot = torch.zeros_like(probs).scatter_(1, pred_label.unsqueeze(1), 1.0)
    pred_bin_np = pred_one_hot.detach().cpu().numpy()
    cc_total = 0.0
    B, C, _, _ = pred_bin_np.shape
    for b in range(B):
        for c in range(C):
            mask = pred_bin_np[b, c]
            if mask.sum() == 0:
                continue
            labeled, num = cc_label(mask)
            if num <= 1:
                continue
            counts = np.bincount(labeled.ravel())
            counts[0] = 0
            largest = counts.argmax() if counts.size > 0 else 0
            largest_mask = (labeled == largest).astype(np.float32)
            largest_mask_t = torch.from_numpy(largest_mask).to(device=y_pred.device)

            extra = pred_one_hot[b, c] * (1.0 - largest_mask_t)
            cc_loss = extra.sum() / (pred_one_hot[b, c].sum() + 1e-7)
            cc_total += weights[c] * cc_loss

    cc_total = cc_total / weights.sum()
    
    # Weighted combination: 70% Dice + 10% Focal + 10% Boundary + 10% CC
    total_loss = 0.7 * loss_dice + 0.10 * loss_focal + 0.1 * loss_boundary + 0.1 * cc_total
    return total_loss

def combined_tversky_focal_loss(y_pred, y_true):
    """Combined Tversky + Focal loss for channels 1 and 2."""
    # y_pred_subset = y_pred[:, 1:3, :, :].contiguous()
    # y_true_subset = y_true[:, 1:3, :, :].contiguous()
    y_pred_subset = y_pred[:, :, :, :].contiguous()
    y_true_subset = y_true[:, :, :, :].contiguous()
    
    tversky_loss = smp.losses.TverskyLoss(mode='multilabel')
    focal_loss = smp.losses.FocalLoss(mode='multilabel')
    
    loss = 0.5 * tversky_loss(y_pred_subset, y_true_subset) + 0.5 * focal_loss(y_pred_subset, y_true_subset)
    return loss


def _probs2one_hot(probs: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
    return (probs > threshold).float()


def _one_hot2hd_dist(seg: np.ndarray) -> np.ndarray:
    """Compute per-class Hausdorff distance transform map.

    Args:
        seg: one-hot array (C, H, W)
    Returns:
        distance maps (C, H, W)
    """
    if seg.ndim != 3:
        raise ValueError("seg must be (C, H, W)")

    C, H, W = seg.shape
    dist = np.zeros((C, H, W), dtype=np.float32)
    diag = np.sqrt((H ** 2) + (W ** 2)).astype(np.float32)
    for c in range(C):
        fg = seg[c].astype(np.bool_)
        if fg.any():
            dist_fg = distance_transform_edt(fg)
            dist_bg = distance_transform_edt(~fg)
            dist[c] = (dist_fg + dist_bg) / (diag + 1e-7)
        else:
            dist[c] = distance_transform_edt(~fg) / (diag + 1e-7)
    return dist


class DiceHausdorffCCLoss(nn.Module):
    """Composite loss: Dice + Hausdorff (boundary) + Connected Components.

    Supports per-class weights for all components.
    """

    def __init__(
        self,
        class_weights: Sequence[float],
        dice_weight: float = 1.0,
        hd_weight: float = 1.0,
        cc_weight: float = 0.2,
        threshold: float = 0.5,
    ) -> None:
        super().__init__()
        self.class_weights = torch.tensor(class_weights, dtype=torch.float32)
        self.dice_weight = dice_weight
        self.hd_weight = hd_weight
        self.cc_weight = cc_weight
        self.threshold = threshold
        self.dice_loss = smp.losses.DiceLoss(mode='multilabel', from_logits=True)
        self.__name__ = "dice_hausdorff_cc_loss"

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute composite loss.

        Args:
            logits: (B, C, H, W) raw outputs
            target: (B, C, H, W) one-hot target
        """
        if logits.shape != target.shape:
            raise ValueError("logits and target must have the same shape")

        device = logits.device
        weights = self.class_weights.to(device=device, dtype=torch.float32)

        # Dice (per-class, weighted)
        dice_total = 0.0
        for c in range(logits.shape[1]):
            dice_total += weights[c] * self.dice_loss(logits[:, c:c+1], target[:, c:c+1])
        dice_total = dice_total / weights.sum()

        # Probabilities for Hausdorff + CC
        probs = torch.sigmoid(logits)

        # Hausdorff distance loss (per-class, weighted)
        hd_total = 0.0
        probs_det = probs.detach().cpu().numpy()
        target_det = target.detach().cpu().numpy()
        B, C, _, _ = probs_det.shape
        for b in range(B):
            target_dm = _one_hot2hd_dist(target_det[b])
            pred_one_hot = _probs2one_hot(torch.from_numpy(probs_det[b]), self.threshold).numpy()
            pred_dm = _one_hot2hd_dist(pred_one_hot)

            delta = (probs_det[b] - target_det[b]) ** 2
            dtm = target_dm ** 2 + pred_dm ** 2
            hd_map = delta * dtm
            for c in range(C):
                hd_total += weights[c] * torch.tensor(hd_map[c].mean(), device=device)
        hd_total = hd_total / (weights.sum() * max(B, 1))

        # Connected component loss (per-class, weighted)
        cc_total = 0.0
        pred_bin = (probs > self.threshold).float()
        pred_bin_np = pred_bin.detach().cpu().numpy()
        for b in range(B):
            for c in range(C):
                mask = pred_bin_np[b, c]
                if mask.sum() == 0:
                    continue
                labeled, num = cc_label(mask)
                if num <= 1:
                    continue
                counts = np.bincount(labeled.ravel())
                counts[0] = 0
                largest = counts.argmax() if counts.size > 0 else 0
                largest_mask = (labeled == largest).astype(np.float32)
                largest_mask_t = torch.from_numpy(largest_mask).to(device=device)

                extra = pred_bin[b, c] * (1.0 - largest_mask_t)
                cc_loss = extra.sum() / (pred_bin[b, c].sum() + 1e-7)
                cc_total += weights[c] * cc_loss

        cc_total = cc_total / weights.sum()

        return (self.dice_weight * dice_total) + (self.hd_weight * hd_total) + (self.cc_weight * cc_total)
    
class MAnetWithDropout(nn.Module):
    """MAnet wrapper that applies spatial dropout on decoder features."""

    def __init__(self, dropout_p=0.2, **manet_kwargs):
        super().__init__()
        self.model = smp.MAnet(**manet_kwargs)
        self.dropout = nn.Dropout2d(p=dropout_p)
        self.dropout_p = dropout_p
        
    def name(self):
        return f"MAnetWithDropout_{self.dropout_p}"

    def forward(self, x):
        features = self.model.encoder(x)
        decoder_output = self.model.decoder(*features)
        decoder_output = self.dropout(decoder_output)
        masks = self.model.segmentation_head(decoder_output)

        if self.model.classification_head is not None:
            labels = self.model.classification_head(features[-1])
            return masks, labels

        return masks
# ----------------

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


def save_loss_plot(train_losses, val_losses, best_epoch, save_path, model_name):
    """Save training and validation loss plot."""
    plt.figure(figsize=(12, 6))
    
    epochs = range(len(train_losses))
    plt.plot(epochs, train_losses, 'r-', label='Training Loss', linewidth=2)
    plt.plot(epochs, val_losses, 'b-', label='Validation Loss', linewidth=2)
    plt.axvline(x=best_epoch, color='g', linestyle='--', 
                label=f'Best Epoch ({best_epoch})', linewidth=2)
    
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.title(f'Training and Validation Loss - {model_name}', fontsize=14, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def train_model(model, train_loader, valid_loader, loss_f, optimizer, scheduler, device, 
                n_epochs=100, display_step=5, save_dir='trained_models', 
                model_name='model', save_fig=False, fig_dir='train_history',
                log_dir='training_logs'):
    """
    Train segmentation model with validation and save best model.
    
    Args:
        model: PyTorch model
        train_loader: Training data loader
        valid_loader: Validation data loader
        loss_f: Loss function
        optimizer: Optimizer
        scheduler: Learning rate scheduler
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
    dsc_history = {f'class_{i}': [] for i in range(N_CLASSES)}
    epoch_times = []
    mean_loss = 0
    best_val_loss = float('inf')
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
    logger.info(f"Model architecture: {model.name()}")
    logger.info(f'Encoder: resnet34')
    logger.info(f"Batch size: {train_loader.batch_size}")
    logger.info(f"Optimizer: {type(optimizer).__name__}")
    logger.info(f"Loss function: {loss_f.__name__}")
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
            loss = loss_f(pred, y)
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
        dsc_accumulator = {f'class_{i}': [] for i in range(N_CLASSES)}
        
        with torch.no_grad():
            for xv, yv in tqdm(valid_loader, desc=f'Epoch {epoch}/{n_epochs} [Valid]'):
                yv = yv.to(device)
                xv = xv.to(device)
                predv = model(xv)
                val_loss = loss_f(predv, yv)
                running_loss += val_loss.item() * xv.size(0)
                predv_np = torch.squeeze(predv.detach().cpu()).numpy()
                predv_argmax = np.argmax(predv_np, axis=0)
                yv_np = torch.squeeze(yv.detach().cpu()).numpy()
                
                # Calculate DSC per class
                dsc_values = calculate_dsc_per_class(predv_argmax, yv_np, num_classes=N_CLASSES)
                for i, dsc in enumerate(dsc_values):
                    dsc_accumulator[f'class_{i}'].append(dsc)
        
        epoch_valid_loss = running_loss / len(valid_loader.dataset)
        plt_val_loss.append(epoch_valid_loss)
        
        # Calculate mean DSC per class
        epoch_dsc = {}
        for class_name, dsc_list in dsc_accumulator.items():
            mean_dsc = np.mean(dsc_list) if dsc_list else 0.0
            epoch_dsc[class_name] = mean_dsc
            dsc_history[class_name].append(mean_dsc)
        
        # Calculate epoch time
        epoch_time = time.time() - epoch_start_time
        epoch_times.append(epoch_time)
        
        # Get current learning rate
        current_lr = optimizer.param_groups[0]['lr']
        
        # Log epoch results with DSC and LR
        dsc_str = ' | '.join([f'{k}: {v:.4f}' for k, v in epoch_dsc.items()])
        logger.info(f"Epoch {epoch}/{n_epochs} - "
                   f"Train Loss: {epoch_train_loss:.5f} | "
                   f"Val Loss: {epoch_valid_loss:.5f} | "
                   f"DSC [{dsc_str}] | "
                   f"LR: {current_lr:.2e} | "
                   f"Time: {epoch_time:.2f}s")
        
        # Save best model with DSC
        if epoch_valid_loss < best_val_loss:
            best_val_loss = epoch_valid_loss
            best_epoch = epoch
            best_model_path = os.path.join(save_dir, f'{model_name}_best.pth')
            torch.save({
                'epoch': epoch,
                'model': model.state_dict(),
                'opt': optimizer.state_dict(),
                'train_loss': epoch_train_loss,
                'val_loss': epoch_valid_loss,
                'dsc': epoch_dsc
            }, best_model_path)
            logger.info(f"✓ Best model saved at epoch {epoch} with val_loss: {epoch_valid_loss:.5f}")
        
        # Update learning rate scheduler
        if scheduler is not None:
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(epoch_valid_loss)
            else:
                scheduler.step()
        
        mean_loss = 0
    
    # Calculate total training time
    total_training_time = time.time() - training_start_time
    avg_epoch_time = np.mean(epoch_times)
    logger.info(f"Total training time: {time.strftime('%H:%M:%S', time.gmtime(total_training_time))}")
    logger.info(f"Average epoch time: {avg_epoch_time:.2f}s")
    
    try: 
        # Final visualization
        print("\nGenerating final visualization...")
        
        # Get final predictions for visualization
        model.eval()
        with torch.no_grad():
            for x, y in valid_loader:
                x = x.to(device)
                y = y.to(device)
                pred = model(x)
                break  # Get first batch only
        
        pred_np = torch_to_np(pred[0].transpose(0, 2).transpose(0, 1))
        x_np = torch_to_np(x[0][0])
        y_np = torch_to_np(y[0].transpose(0, 2).transpose(0, 1))
        
        # Create comprehensive visualization
        fig = plt.figure(figsize=(20, 12))
        
        # Plot 1: Training and Validation Loss
        plt.subplot(3, 3, 1)
        plt.plot(plt_loss, 'r-', linewidth=2, label='Train Loss')
        plt.plot(plt_val_loss, 'b-', linewidth=2, label='Val Loss')
        plt.axvline(x=best_epoch, color='g', linestyle='--', linewidth=2, label=f'Best Epoch ({best_epoch})')
        plt.legend()
        plt.title('Loss Curves', fontweight='bold')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.grid(True, alpha=0.3)
        
        # Plot 2-4: DSC per class
        for i in range(N_CLASSES):
            plt.subplot(3, 3, i + 2)
            plt.plot(dsc_history[f'class_{i}'], linewidth=2, color=['red', 'green', 'blue'][i])
            plt.axvline(x=best_epoch, color='g', linestyle='--', linewidth=1, alpha=0.5)
            plt.title(f'DSC - Class {i}', fontweight='bold')
            plt.xlabel('Epoch')
            plt.ylabel('DSC')
            plt.grid(True, alpha=0.3)
            plt.ylim([0, 1])
        
        # Plot 5: Input Image
        plt.subplot(3, 3, 5)
        plt.imshow(x_np, cmap='gray')
        plt.title('Input Image', fontweight='bold')
        plt.axis('off')
        
        # Plot 6: Ground Truth
        plt.subplot(3, 3, 6)
        plt.imshow(y_np)
        plt.title('Ground Truth', fontweight='bold')
        plt.axis('off')
        
        # Plot 7: Prediction
        plt.subplot(3, 3, 7)
        plt.imshow(pred_np)
        plt.title('Prediction', fontweight='bold')
        plt.axis('off')
        
        # Plot 8: Overlay
        plt.subplot(3, 3, 8)
        plt.imshow(x_np, cmap='gray')
        plt.imshow(pred_np, alpha=0.5)
        plt.title('Prediction Overlay', fontweight='bold')
        plt.axis('off')
        
        # Plot 9: Summary metrics as text
        plt.subplot(3, 3, 9)
        plt.axis('off')
    except Exception as e:
        logger.error(f"Error during final visualization: {e}")
        pass
    
    summary_text = f"""Training Summary
    
Total Epochs: {n_epochs}
Best Epoch: {best_epoch}

Best Val Loss: {best_val_loss:.5f}
Final Train Loss: {plt_loss[-1]:.5f}
Final Val Loss: {plt_val_loss[-1]:.5f}

Final DSC:
"""
    for i in range(N_CLASSES):
        summary_text += f"  Class {i}: {dsc_history[f'class_{i}'][-1]:.4f}\n"
    
    summary_text += f"\nTraining Time: {time.strftime('%H:%M:%S', time.gmtime(total_training_time))}"
    plt.text(0.1, 0.5, summary_text, fontsize=11, verticalalignment='center', 
             family='monospace', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
    
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
        'val_loss': epoch_valid_loss
    }, final_model_path)
    
    # Save training history
    history = {
        'train_loss': [float(loss) for loss in plt_loss],
        'val_loss': [float(loss) for loss in plt_val_loss],
        'dsc_per_class': {k: [float(v) for v in vals] for k, vals in dsc_history.items()},
        'epoch_times': [float(t) for t in epoch_times],
        'best_epoch': int(best_epoch),
        'best_val_loss': float(best_val_loss),
        'total_training_time': float(total_training_time),
        'avg_epoch_time': float(avg_epoch_time)
    }
    history_path = os.path.join(log_dir, f'{model_name}_history.json')
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=4)
    
    # Save final loss plot
    loss_plot_path = os.path.join(log_dir, f'{model_name}_loss_plot.png')
    save_loss_plot(plt_loss, plt_val_loss, best_epoch, loss_plot_path, model_name)
    
    # Log training summary
    logger.info("="*70)
    logger.info("TRAINING COMPLETE")
    logger.info("="*70)
    logger.info(f"Total training time: {time.strftime('%H:%M:%S', time.gmtime(total_training_time))}")
    logger.info(f"Average epoch time: {avg_epoch_time:.2f}s")
    logger.info(f"Best epoch: {best_epoch}")
    logger.info(f"Best validation loss: {best_val_loss:.5f}")
    logger.info(f"Final training loss: {plt_loss[-1]:.5f}")
    logger.info(f"Final validation loss: {plt_val_loss[-1]:.5f}")
    logger.info(f"Final model saved to: {final_model_path}")
    logger.info(f"Best model saved to: {best_model_path}")
    logger.info(f"Training history saved to: {history_path}")
    logger.info(f"Loss plot saved to: {loss_plot_path}")
    logger.info(f"Training log saved to: {log_file}")
    logger.info("="*70)
    
    return plt_loss, plt_val_loss, best_epoch, best_val_loss


def train_segmentation_model(organ='rib', train_data_dir='Dataset/train',
                             valid_data_dir='Dataset/validation',
                             train_batch_size=4, n_epochs=100, display_step=5,
                             encoder='resnet34', encoder_weights='imagenet',
                             learning_rate=0.0001, save_dir='trained_models',
                             save_fig=False, fig_dir='train_history',
                             log_dir='training_logs', dropout_p=0.2,
                             class_weights=(0.01, 1.0, 1.0),
                             dice_weight=1.0, hd_weight=1.0, cc_weight=0.2,
                             cc_threshold=0.5):
    """
    Main function to train segmentation model for rib or lung.
    
    Args:
        organ: 'rib' or 'lung'
        train_data_dir: Directory containing training data with full/not full subfolders
        valid_data_dir: Directory containing validation data with full/not full subfolders
        train_batch_size: Batch size for training
        n_epochs: Number of training epochs
        display_step: Display frequency
        encoder: Encoder architecture
        encoder_weights: Pretrained weights
        learning_rate: Learning rate
        save_dir: Directory to save trained models
        save_fig: Whether to save training figures
        fig_dir: Directory to save figures
        log_dir: Directory to save training logs
    """
    print(f"\n{'='*70}")
    print(f"TRAINING {organ.upper()} SEGMENTATION MODEL")
    print(f"{'='*70}\n")
    
    # Load data from folders
    print("Loading data from folders...")
    
    # Training data - combine full and not full
    print("Loading training images...")
    train_full_images, train_full_paths = load_image(os.path.join(train_data_dir, 'full/image'))
    train_notfull_images, train_notfull_paths = load_image(os.path.join(train_data_dir, 'not full/image'))
    train_images = np.concatenate([train_full_images, train_notfull_images], axis=0)
    
    print(f"Loading training {organ} masks...")
    train_full_masks, train_full_mask_paths = load_mask(os.path.join(train_data_dir, f'full/mask/{organ}'))
    train_notfull_masks, train_notfull_mask_paths = load_mask(os.path.join(train_data_dir, f'not full/mask/{organ}'))
    train_masks = np.concatenate([train_full_masks, train_notfull_masks], axis=0)
    
    # Validation data - combine full and not full
    print("Loading validation images...")
    valid_full_images, valid_full_paths = load_image(os.path.join(valid_data_dir, 'full/image'))
    valid_notfull_images, valid_notfull_paths = load_image(os.path.join(valid_data_dir, 'not full/image'))
    valid_images = np.concatenate([valid_full_images, valid_notfull_images], axis=0)
    
    print(f"Loading validation {organ} masks...")
    valid_full_masks, valid_full_mask_paths = load_mask(os.path.join(valid_data_dir, f'full/mask/{organ}'))
    valid_notfull_masks, valid_notfull_mask_paths = load_mask(os.path.join(valid_data_dir, f'not full/mask/{organ}'))
    valid_masks = np.concatenate([valid_full_masks, valid_notfull_masks], axis=0)
    
    print(f"\nTrain images: {train_images.shape}")
    print(f"Train masks: {train_masks.shape}")
    print(f"Valid images: {valid_images.shape}")
    print(f"Valid masks: {valid_masks.shape}")
    print(f"Unique mask values: {np.unique(train_masks)}\n")
    
    # Create datasets
    ENCODER = encoder
    ENCODER_WEIGHTS = encoder_weights
    preprocessing_fn = smp.encoders.get_preprocessing_fn(ENCODER, ENCODER_WEIGHTS)
    train_dataset = PreprocessedDataset(
        train_images, train_masks, 
        augmentation=get_training_augmentation(),
        preprocess=preprocessing_fn
    )
    valid_dataset = PreprocessedDataset(
        valid_images, valid_masks,
        augmentation=None,
        preprocess=preprocessing_fn
    )
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=1, shuffle=False)
    
    # Setup device (with MPS support for Mac)
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
    # elif torch.backends.mps.is_available():
    #     device = torch.device("mps")
    else:
        device = torch.device("cpu")
    
    print(f"Using device: {device}")
        
    
    # Create model
    
    print(f"Creating model: MAnet with {encoder} (dropout_p={dropout_p})")
    model = MAnetWithDropout(
        dropout_p=dropout_p,
        encoder_name=encoder,
        encoder_weights=encoder_weights,
        in_channels=3,
        classes=N_CLASSES
    )
    
    # Move to device and optimize for MPS if available
    model.to(device)
    if device.type == 'mps':
        model = model.to(memory_format=torch.channels_last)
    
    # ------------------------------- #
    # Setup loss functions
    # ------------------------------- #
    # [Option 1] Subset Dice Loss (channels 1 and 2 only)
    # loss_f = sub_dice_loss
    # loss_name = 'sub_dice_loss'
    # [Option 2] Combined BCE + Dice Loss
    # loss_f = combined_bce_dice_loss
    # loss_name = 'combined_bce_dice_loss'
    # [Option 3] Focal + Dice Loss
    # loss_f = focal_dice_loss
    # loss_name = 'focal_dice_loss'
    # [Option 4] Combined Tversky + Focal Loss
    # loss_f = combined_tversky_focal_loss
    # loss_name = 'combined_tversky_focal_loss'
    # [Option 5] Boundary + Dice Loss
    # loss_f = boundary_dice_loss
    # loss_name = 'boundary_dice_loss'
    # [Option 6] Boundary + Focal + Dice + CC Loss (recommended)
    loss_f = boundary_focal_dice_cc_loss
    loss_name = 'boundary_focal_dice_cc_loss'
    # [Option 7] Combined Dice + BCE + Inverse Dice Loss
    # loss_f = dice_bce_inverse_loss_3channel
    # loss_name = 'dice_bce_inverse_loss_3channel'
    # [Option 8] Weighted Dice Loss
    # loss_f = weighted_dice_loss
    # loss_name = 'weighted_dice_loss'
    # [Option 9] Dice + Hausdorff + Connected Component Loss
    # loss_f = DiceHausdorffCCLoss(
    #     class_weights=class_weights,
    #     dice_weight=dice_weight,
    #     hd_weight=hd_weight,
    #     cc_weight=cc_weight,
    #     threshold=cc_threshold
    # )
    # loss_name = loss_f.__name__

    print("----------------------------------")
    print(f"Using loss function: {loss_name}")
    print("----------------------------------\n")
    
    # Setup optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    
    # Setup learning rate scheduler
    # Option 1: ReduceLROnPlateau - reduces LR when validation loss plateaus (RECOMMENDED)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, verbose=True, min_lr=1e-7
    )
    
    # Option 2: CosineAnnealingLR - smooth cosine decay
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    #     optimizer, T_max=n_epochs, eta_min=1e-7
    # )
    
    # CosineAnnealingWarmRestarts
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
    #     optimizer, T_0=10, T_mult=2, eta_min=1e-7
    # )
    
    # Option 3: StepLR - step decay every N epochs
    # scheduler = torch.optim.lr_scheduler.StepLR(
    #     optimizer, step_size=30, gamma=0.5
    # )
    
    # Option 4: OneCycleLR - one cycle policy (good for fast training)
    # scheduler = torch.optim.lr_scheduler.OneCycleLR(
    #     optimizer, max_lr=learning_rate*10, 
    #     steps_per_epoch=len(train_loader), epochs=n_epochs
    # )
    
    # Option 5: No scheduler
    # scheduler = None
    
    
    
    print(f"Using scheduler: {type(scheduler).__name__ if scheduler is not None else 'None'}\n")
    
    # Model name
    model_name = f'{organ}_MAnet_{datetime.now().strftime("%Y%m%d_%H%M%S")}'
    
    # Train model
    train_loss, val_loss, best_epoch, best_val_loss = train_model(
        model=model,
        train_loader=train_loader,
        valid_loader=valid_loader,
        loss_f=loss_f,
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
    
    return model, train_loss, val_loss


def main():
    """Main function for command-line usage."""
    parser = argparse.ArgumentParser(description='Train segmentation models')
    parser.add_argument('--organ', type=str, default='rib', choices=['rib', 'lung'],
                       help='Organ to train model for (rib or lung)')
    parser.add_argument('--train_data_dir', type=str, default='../2026 Dataset/train',
                       help='Directory containing training data with full/not full subfolders')
    parser.add_argument('--valid_data_dir', type=str, default='../2026 Dataset/validation',
                       help='Directory containing validation data with full/not full subfolders')
    parser.add_argument('--batch_size', type=int, default=4,
                       help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=100,
                       help='Number of training epochs')
    parser.add_argument('--display_step', type=int, default=5,
                       help='Display frequency')
    parser.add_argument('--encoder', type=str, default='resnet34',
                       help='Encoder architecture')
    parser.add_argument('--encoder_weights', type=str, default='imagenet',
                       help='Pretrained encoder weights')
    parser.add_argument('--lr', type=float, default=0.0001,
                       help='Learning rate')
    parser.add_argument('--dropout_p', type=float, default=0.4,
                       help='Spatial dropout probability in decoder')
    parser.add_argument('--class_weights', type=float, nargs=3, default=(0.00001, 1.0, 1.0),
                       help='Per-class weights (C0 C1 C2)')
    parser.add_argument('--dice_weight', type=float, default=0.3,
                       help='Weight for Dice term in composite loss')
    parser.add_argument('--hd_weight', type=float, default=0.4,
                       help='Weight for Hausdorff term in composite loss')
    parser.add_argument('--cc_weight', type=float, default=0.2,
                       help='Weight for connected-component term in composite loss')
    parser.add_argument('--cc_threshold', type=float, default=0.5,
                       help='Threshold for CC loss binarization')
    parser.add_argument('--save_dir', type=str, default='../train_result/trained_models',
                       help='Directory to save trained models')
    parser.add_argument('--save_fig', action='store_true',
                       help='Save training figures')
    parser.add_argument('--fig_dir', type=str, default='../train_result/train_history',
                       help='Directory to save training figures')
    parser.add_argument('--log_dir', type=str, default='../train_result/training_logs',
                       help='Directory to save training logs')
    parser.add_argument('--train_both', action='store_true',
                       help='Train both rib and lung models')
    
    args = parser.parse_args()
    
    if args.train_both:
        # Train both models
        print("\n" + "="*70)
        print("TRAINING BOTH RIB AND LUNG MODELS")
        print("="*70 + "\n")
        
        for organ in ['rib','lung']:
            train_segmentation_model(
                organ=organ,
                train_data_dir=args.train_data_dir,
                valid_data_dir=args.valid_data_dir,
                train_batch_size=args.batch_size,
                n_epochs=args.epochs,
                display_step=args.display_step,
                encoder=args.encoder,
                encoder_weights=args.encoder_weights,
                learning_rate=args.lr,
                save_dir=args.save_dir,
                save_fig=args.save_fig,
                fig_dir=args.fig_dir,
                log_dir=args.log_dir,
                dropout_p=args.dropout_p,
                class_weights=tuple(args.class_weights),
                dice_weight=args.dice_weight,
                hd_weight=args.hd_weight,
                cc_weight=args.cc_weight,
                cc_threshold=args.cc_threshold
            )
    else:
        # Train single model
        train_segmentation_model(
            organ=args.organ,
            train_data_dir=args.train_data_dir,
            valid_data_dir=args.valid_data_dir,
            train_batch_size=args.batch_size,
            n_epochs=args.epochs,
            display_step=args.display_step,
            encoder=args.encoder,
            encoder_weights=args.encoder_weights,
            learning_rate=args.lr,
            save_dir=args.save_dir,
            save_fig=args.save_fig,
            fig_dir=args.fig_dir,
            log_dir=args.log_dir,
            dropout_p=args.dropout_p,
            class_weights=tuple(args.class_weights),
            dice_weight=args.dice_weight,
            hd_weight=args.hd_weight,
            cc_weight=args.cc_weight,
            cc_threshold=args.cc_threshold
        )


if __name__ == '__main__':
    print("="*70)
    print("SEGMENTATION MODEL TRAINING SCRIPT")
    print("="*70)
    print("\nUsage examples:")
    print("\n1. Train rib model:")
    print("   python train_segment.py --organ rib")
    print("\n2. Train lung model:")
    print("   python train_segment.py --organ lung")
    print("\n3. Train both models:")
    print("   python train_segment.py --train_both")
    print("\n4. Custom training:")
    print("   python train_segment.py --organ rib --epochs 50 --batch_size 8 --save_fig")
    print("\n" + "="*70 + "\n")
    
    main()
