import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Tuple
from dataclasses import dataclass
import logging
from tqdm import tqdm
import time
from collections import Counter

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.metrics import (classification_report, confusion_matrix,
                            accuracy_score, precision_recall_fscore_support,
                            roc_auc_score, roc_curve)
import seaborn as sns

@dataclass
class Config:
    TRAIN_PATH: str = '/content/drive/MyDrive/archive/training'
    TEST_PATH: str = '/content/drive/MyDrive/archive/testing'

    IMG_SIZE: Tuple[int, int] = (224, 224)
    RANDOM_STATE: int = 42

    BATCH_SIZE: int = 32
    NUM_EPOCHS: int = 10
    LEARNING_RATE: float = 1e-4
    WEIGHT_DECAY: float = 1e-4

    NUM_WORKERS: int = 0
    PIN_MEMORY: bool = False

    BACKGROUND_THRESHOLD: int = 10
    MORPH_KERNEL_SIZE: Tuple[int, int] = (3, 3)
    GAUSSIAN_SIGMA: float = 80.0
    DENOISE_H: int = 5
    CLAHE_CLIP_LIMIT: float = 2.5
    CLAHE_TILE_SIZE: Tuple[int, int] = (8, 8)

    RANDOM_ERASING_PROB: float = 0.2
    RANDOM_ERASING_SL: float = 0.020
    RANDOM_ERASING_SH: float = 0.4
    ROTATION_RANGE: Tuple[float, float] = (-50, 50)
    BRIGHTNESS_RANGE: Tuple[float, float] = (0.50, 1.50)
    CONTRAST_RANGE: Tuple[float, float] = (0.50, 1.50)

    NUM_SAMPLE_IMAGES: int = 3
    VISUALIZE_STEPS: bool = True

def setup_logging(log_level=logging.INFO) -> logging.Logger:
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    return logging.getLogger(__name__)

logger = setup_logging()

class MRIPreprocessor:
    def __init__(self, config: Config):
        self.config = config
        self.preprocessing_stats = {
            'background_removed': 0,
            'bias_corrected': 0,
            'denoised': 0,
            'normalized': 0,
            'contrast_enhanced': 0,
            'sharpened': 0
        }

    def remove_background(self, img: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        kernel = np.ones(self.config.MORPH_KERNEL_SIZE, np.uint8)
        cleaned = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=2)
        cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, kernel, iterations=1)

        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(cleaned, connectivity=8)
        if num_labels > 1:
            largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
            cleaned = np.where(labels == largest_label, 255, 0).astype(np.uint8)

        mask_3ch = cv2.cvtColor(cleaned, cv2.COLOR_GRAY2RGB)
        self.preprocessing_stats['background_removed'] += 1
        return cv2.bitwise_and(img, mask_3ch)

    def bias_field_correction(self, img: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY).astype(np.float32)
        mask = gray > 0

        if not np.any(mask):
            return img

        blurred = cv2.GaussianBlur(gray, (0, 0), sigmaX=self.config.GAUSSIAN_SIGMA)
        blurred[blurred == 0] = 1

        mean_intensity = np.mean(gray[mask])
        corrected = gray / (blurred / mean_intensity)
        corrected = np.clip(corrected, 0, 255).astype(np.uint8)

        self.preprocessing_stats['bias_corrected'] += 1
        return cv2.cvtColor(corrected, cv2.COLOR_GRAY2RGB)

    def denoise_advanced(self, img: np.ndarray) -> np.ndarray:
        img_yuv = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
        img_yuv[:, :, 0] = cv2.fastNlMeansDenoising(
            img_yuv[:, :, 0], None, h=self.config.DENOISE_H
        )
        self.preprocessing_stats['denoised'] += 1
        return cv2.cvtColor(img_yuv, cv2.COLOR_YUV2RGB)

    def normalize_intensity(self, img: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        mask = gray > 0

        if not np.any(mask):
            return img

        gray_normalized = gray.copy().astype(np.float32)
        gray_normalized[mask] = ((gray[mask] - gray[mask].min()) /
                                (gray[mask].max() - gray[mask].min()) * 255)

        self.preprocessing_stats['normalized'] += 1
        return cv2.cvtColor(gray_normalized.astype(np.uint8), cv2.COLOR_GRAY2RGB)

    def enhance_contrast(self, img: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        clahe = cv2.createCLAHE(
            clipLimit=self.config.CLAHE_CLIP_LIMIT,
            tileGridSize=self.config.CLAHE_TILE_SIZE
        )
        enhanced = clahe.apply(gray)
        self.preprocessing_stats['contrast_enhanced'] += 1
        return cv2.cvtColor(enhanced, cv2.COLOR_GRAY2RGB)

    def sharpen_edges(self, img: np.ndarray) -> np.ndarray:
        gaussian = cv2.GaussianBlur(img, (0, 0), 2.0)
        sharpened = cv2.addWeighted(img, 1.5, gaussian, -0.5, 0)
        self.preprocessing_stats['sharpened'] += 1
        return np.clip(sharpened, 0, 255).astype(np.uint8)

    def process_pipeline(self, img: np.ndarray, return_steps: bool = False) -> np.ndarray:
        if return_steps:
            steps = {'original': img.copy()}

            img = self.remove_background(img)
            steps['background_removed'] = img.copy()

            img = self.bias_field_correction(img)
            steps['bias_corrected'] = img.copy()

            img = self.denoise_advanced(img)
            steps['denoised'] = img.copy()

            img = self.normalize_intensity(img)
            steps['normalized'] = img.copy()

            img = self.enhance_contrast(img)
            steps['contrast_enhanced'] = img.copy()

            img = self.sharpen_edges(img)
            steps['sharpened'] = img.copy()

            return img, steps
        else:
            img = self.remove_background(img)
            img = self.bias_field_correction(img)
            img = self.denoise_advanced(img)
            img = self.normalize_intensity(img)
            img = self.enhance_contrast(img)
            img = self.sharpen_edges(img)
            return img

class MRIDataset(Dataset):
    def __init__(self, image_paths: List[Path], labels: List[int],
                 config: Config, preprocessor: MRIPreprocessor,
                 mode: str = 'train'):
        self.image_paths = image_paths
        self.labels = labels
        self.config = config
        self.preprocessor = preprocessor
        self.mode = mode

        self.normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        img_path = self.image_paths[idx]
        img = cv2.imread(str(img_path))

        if img is None:
            raise ValueError(f"Failed to load image: {img_path}")

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = self.preprocessor.process_pipeline(img)
        img = cv2.resize(img, self.config.IMG_SIZE)

        if self.mode == 'train':
            img = self._apply_augmentation(img)

        img = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
        img = self.normalize(img)

        return img, self.labels[idx]

    def _apply_augmentation(self, img: np.ndarray) -> np.ndarray:
        if np.random.rand() > 0.5:
            img = cv2.flip(img, 1)

        if np.random.rand() > 0.5:
            angle = np.random.uniform(*self.config.ROTATION_RANGE)
            h, w = img.shape[:2]
            M = cv2.getRotationMatrix2D((w/2, h/2), angle, 1.0)
            img = cv2.warpAffine(img, M, (w, h), borderValue=(0, 0, 0))

        if np.random.rand() > 0.5:
            factor = np.random.uniform(*self.config.BRIGHTNESS_RANGE)
            img = np.clip(img.astype(np.float32) * factor, 0, 255).astype(np.uint8)

        if np.random.rand() > 0.5:
            factor = np.random.uniform(*self.config.CONTRAST_RANGE)
            mean = img.mean()
            img = np.clip((img - mean) * factor + mean, 0, 255).astype(np.uint8)

        if np.random.rand() < self.config.RANDOM_ERASING_PROB:
            img = self._random_erasing(img)

        return img

    def _random_erasing(self, img: np.ndarray) -> np.ndarray:
        h, w = img.shape[:2]
        area = h * w
        num_erasures = np.random.randint(3, 6)

        for _ in range(num_erasures):
            target_area = np.random.uniform(
                self.config.RANDOM_ERASING_SL,
                self.config.RANDOM_ERASING_SH
            ) * area

            patch_h = int(np.sqrt(target_area))
            patch_w = int(np.sqrt(target_area))

            if patch_h < h and patch_w < w:
                x = np.random.randint(0, w - patch_w)
                y = np.random.randint(0, h - patch_h)
                strength = np.random.uniform(0.5, 0.9)
                img[y:y+patch_h, x:x+patch_w] = \
                    (img[y:y+patch_h, x:x+patch_w] * (1 - strength)).astype(np.uint8)

        return img

class DataVisualizer:
    def __init__(self, config: Config):
        self.config = config

    def visualize_preprocessing_steps(self, sample_images: List[np.ndarray],
                                     preprocessor: MRIPreprocessor,
                                     class_names: List[str],
                                     sample_labels: List[int]):

        n_samples = len(sample_images)
        n_steps = 7

        fig, axes = plt.subplots(n_samples, n_steps, figsize=(n_steps*3, n_samples*3))
        if n_samples == 1:
            axes = axes.reshape(1, -1)

        step_names = ['Original', 'Background Removed', 'Bias Corrected',
                     'Denoised', 'Normalized', 'Contrast Enhanced', 'Sharpened']

        for idx, (img, label) in enumerate(zip(sample_images, sample_labels)):
            _, steps = preprocessor.process_pipeline(img, return_steps=True)

            for step_idx, step_name in enumerate(step_names):
                ax = axes[idx, step_idx]
                step_key = step_name.lower().replace(' ', '_')
                ax.imshow(steps[step_key])
                ax.axis('off')
                if idx == 0:
                    ax.set_title(step_name, fontsize=10, fontweight='bold')
                if step_idx == 0:
                    ax.text(-0.1, 0.5, f'{class_names[label]}',
                           transform=ax.transAxes, rotation=90,
                           va='center', fontsize=10, fontweight='bold')

        plt.tight_layout()
        plt.show()
        logger.info("✓ Preprocessing steps visualization displayed")

    def visualize_augmentation_examples(self, sample_image: np.ndarray,
                                       class_name: str):
        logger.info("\n Displaying augmentation examples...")

        np.random.seed(self.config.RANDOM_STATE)

        augmentations = {
            'Original': sample_image.copy(),
            'Horizontal Flip': cv2.flip(sample_image, 1),
            'Rotation (+50°)': self._rotate_image(sample_image, 50),
            'Rotation (-50°)': self._rotate_image(sample_image, -50),
            'Brightness ↑': self._adjust_brightness(sample_image, 1.50),
            'Brightness ↓': self._adjust_brightness(sample_image, 0.50),
            'Contrast ↑': self._adjust_contrast(sample_image, 1.50),
            'Contrast ↓': self._adjust_contrast(sample_image, 0.50),
            'Random Erasing': self._apply_random_erasing(sample_image.copy())
        }

        fig, axes = plt.subplots(3, 3, figsize=(12, 12))
        axes = axes.flatten()

        for idx, (aug_name, aug_img) in enumerate(augmentations.items()):
            axes[idx].imshow(aug_img)
            axes[idx].set_title(aug_name, fontsize=10, fontweight='bold')
            axes[idx].axis('off')

        plt.suptitle(f'Data Augmentation Examples - {class_name}',
                    fontsize=12, fontweight='bold', y=0.995)
        plt.tight_layout()
        plt.show()
        logger.info("✓ Augmentation examples visualization displayed")

    def _rotate_image(self, img: np.ndarray, angle: float) -> np.ndarray:
        h, w = img.shape[:2]
        M = cv2.getRotationMatrix2D((w/2, h/2), angle, 1.0)
        return cv2.warpAffine(img, M, (w, h), borderValue=(0, 0, 0))

    def _adjust_brightness(self, img: np.ndarray, factor: float) -> np.ndarray:
        return np.clip(img.astype(np.float32) * factor, 0, 255).astype(np.uint8)

    def _adjust_contrast(self, img: np.ndarray, factor: float) -> np.ndarray:
        mean = img.mean()
        return np.clip((img - mean) * factor + mean, 0, 255).astype(np.uint8)

    def _apply_random_erasing(self, img: np.ndarray) -> np.ndarray:
        h, w = img.shape[:2]
        area = h * w

        for _ in range(4):
            target_area = 0.08 * area
            patch_h = int(np.sqrt(target_area))
            patch_w = int(np.sqrt(target_area))

            if patch_h < h and patch_w < w:
                x = np.random.randint(0, w - patch_w)
                y = np.random.randint(0, h - patch_h)
                img[y:y+patch_h, x:x+patch_w] = \
                    (img[y:y+patch_h, x:x+patch_w] * 0.3).astype(np.uint8)

        return img

    def plot_dataset_distribution(self, train_labels: List[int],
                                  test_labels: List[int],
                                  class_names: List[str]):
        logger.info("\n Displaying dataset distribution...")

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        train_counts = Counter(train_labels)
        axes[0].bar(range(len(class_names)),
                   [train_counts[i] for i in range(len(class_names))],
                   color='steelblue', edgecolor='black')
        axes[0].set_xticks(range(len(class_names)))
        axes[0].set_xticklabels(class_names, rotation=45, ha='right')
        axes[0].set_ylabel('Number of Images', fontsize=11)
        axes[0].set_title('Training Set Distribution', fontsize=12, fontweight='bold')
        axes[0].grid(axis='y', alpha=0.3)

        for i, count in enumerate([train_counts[i] for i in range(len(class_names))]):
            axes[0].text(i, count + 50, str(count), ha='center', fontweight='bold')

        test_counts = Counter(test_labels)
        axes[1].bar(range(len(class_names)),
                   [test_counts[i] for i in range(len(class_names))],
                   color='coral', edgecolor='black')
        axes[1].set_xticks(range(len(class_names)))
        axes[1].set_xticklabels(class_names, rotation=45, ha='right')
        axes[1].set_ylabel('Number of Images', fontsize=11)
        axes[1].set_title('Test Set Distribution', fontsize=12, fontweight='bold')
        axes[1].grid(axis='y', alpha=0.3)

        for i, count in enumerate([test_counts[i] for i in range(len(class_names))]):
            axes[1].text(i, count + 10, str(count), ha='center', fontweight='bold')

        plt.tight_layout()
        plt.show()
        logger.info("✓ Dataset distribution displayed")

class ModelFactory:
    @staticmethod
    def create_model(model_name: str, num_classes: int, pretrained: bool = True) -> nn.Module:
        model_name = model_name.lower()

        try:
            if model_name == 'resnet-50':
                model = models.resnet50(pretrained=pretrained)
                num_features = model.fc.in_features
                model.fc = nn.Sequential(
                    nn.Dropout(0.5),
                    nn.Linear(num_features, num_classes)
                )

            elif model_name == 'resnet-101':
                model = models.resnet101(pretrained=pretrained)
                num_features = model.fc.in_features
                model.fc = nn.Sequential(
                    nn.Dropout(0.5),
                    nn.Linear(num_features, num_classes)
                )

            elif model_name == 'resnet-18':
                model = models.resnet18(pretrained=pretrained)
                num_features = model.fc.in_features
                model.fc = nn.Sequential(
                    nn.Dropout(0.5),
                    nn.Linear(num_features, num_classes)
                )

            elif model_name == 'efficientnet-b0':
                model = models.efficientnet_b0(pretrained=pretrained)
                num_features = model.classifier[1].in_features
                model.classifier[1] = nn.Linear(num_features, num_classes)

            elif model_name == 'mobilenet-v2':
                model = models.mobilenet_v2(pretrained=pretrained)
                num_features = model.classifier[1].in_features
                model.classifier[1] = nn.Linear(num_features, num_classes)

            elif model_name == 'densenet-201':
                model = models.densenet201(pretrained=pretrained)
                num_features = model.classifier.in_features
                model.classifier = nn.Linear(num_features, num_classes)

            elif model_name == 'vgg-19':
                model = models.vgg19(pretrained=pretrained)
                num_features = model.classifier[6].in_features
                model.classifier[6] = nn.Linear(num_features, num_classes)

            elif model_name == 'vgg-16':
                model = models.vgg16(pretrained=pretrained)
                num_features = model.classifier[6].in_features
                model.classifier[6] = nn.Linear(num_features, num_classes)

            else:
                raise ValueError(f"Model '{model_name}' not supported!")

            logger.info(f"✓ {model_name.upper()} model created")
            return model

        except Exception as e:
            logger.error(f"Error creating {model_name}: {str(e)}")
            raise

class ModelTrainer:
    def __init__(self, model: nn.Module, model_name: str, config: Config,
                 train_loader: DataLoader, class_names: List[str]):
        self.model = model
        self.model_name = model_name
        self.config = config
        self.train_loader = train_loader
        self.class_names = class_names

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self.model.to(self.device)

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=config.LEARNING_RATE,
            weight_decay=config.WEIGHT_DECAY
        )

        self.scheduler = ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=5
        )

        self.history = {
            'train_loss': [],
            'train_acc': [],
            'lr': []
        }

        self.best_train_loss = float('inf')
        self.training_time = 0

    def train_epoch(self) -> Tuple[float, float]:
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        pbar = tqdm(self.train_loader, desc=f'{self.model_name} Training')
        for inputs, labels in pbar:
            inputs, labels = inputs.to(self.device), labels.to(self.device)

            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.criterion(outputs, labels)

            loss.backward()
            self.optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{100. * correct / total:.2f}%'
            })

        epoch_loss = running_loss / total
        epoch_acc = correct / total

        return epoch_loss, epoch_acc

    def train(self) -> Dict:
        logger.info("="*70)
        logger.info(f"STARTING {self.model_name.upper()} TRAINING")
        logger.info("="*70)

        start_time = time.time()

        for epoch in range(self.config.NUM_EPOCHS):
            train_loss, train_acc = self.train_epoch()

            self.scheduler.step(train_loss)
            current_lr = self.optimizer.param_groups[0]['lr']

            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            self.history['lr'].append(current_lr)

            if train_loss < self.best_train_loss:
                self.best_train_loss = train_loss

        self.training_time = time.time() - start_time

        logger.info(f"✓ {self.model_name.upper()} COMPLETED - Time: {self.training_time:.2f}s")

        self._display_training_curves()

        return self.history

    def _display_training_curves(self):
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))

        axes[0].plot(self.history['train_loss'], label='Train Loss', linewidth=2)
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].set_title(f'{self.model_name.upper()} - Loss Curve')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        axes[1].plot(self.history['train_acc'], label='Train Acc', linewidth=2)
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Accuracy')
        axes[1].set_title(f'{self.model_name.upper()} - Accuracy Curve')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()
        logger.info(f"✓ {self.model_name} training curves displayed")

class ModelEvaluator:
    def __init__(self, model: nn.Module, model_name: str,
                 test_loader: DataLoader, class_names: List[str],
                 device: torch.device):
        self.model = model
        self.model_name = model_name
        self.test_loader = test_loader
        self.class_names = class_names
        self.device = device

    def evaluate(self) -> Dict:
        self.model.eval()
        all_preds = []
        all_labels = []
        all_probs = []

        with torch.no_grad():
            for inputs, labels in tqdm(self.test_loader, desc=f'{self.model_name} Testing'):
                inputs = inputs.to(self.device)
                outputs = self.model(inputs)
                probs = torch.softmax(outputs, dim=1)
                _, preds = torch.max(outputs, 1)

                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.numpy())
                all_probs.extend(probs.cpu().numpy())

        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)
        all_probs = np.array(all_probs)

        accuracy = accuracy_score(all_labels, all_preds)
        precision, recall, f1, _ = precision_recall_fscore_support(
            all_labels, all_preds, average='weighted', zero_division=0
        )

        results = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'predictions': all_preds,
            'labels': all_labels,
            'probabilities': all_probs
        }

        self._display_confusion_matrix(all_labels, all_preds)
        self._display_roc_curves(all_labels, all_probs)

        return results

    def _display_confusion_matrix(self, labels, preds):
        cm = confusion_matrix(labels, preds)

        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=self.class_names,
                   yticklabels=self.class_names)
        plt.title(f'{self.model_name.upper()} - Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.show()
        logger.info(f"✓ {self.model_name} confusion matrix displayed")

    def _display_roc_curves(self, labels, probs):
        n_classes = len(self.class_names)

        plt.figure(figsize=(10, 8))

        for i in range(n_classes):
            y_true = (labels == i).astype(int)
            y_score = probs[:, i]

            fpr, tpr, _ = roc_curve(y_true, y_score)
            auc = roc_auc_score(y_true, y_score)

            plt.plot(fpr, tpr, linewidth=2,
                    label=f'{self.class_names[i]} (AUC = {auc:.3f})')

        plt.plot([0, 1], [0, 1], 'k--', linewidth=2, label='Random')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'{self.model_name.upper()} - ROC Curves')
        plt.legend(loc="lower right")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
        logger.info(f"✓ {self.model_name} ROC curves displayed")

class MultiModelComparison:
    def __init__(self, results: Dict[str, Dict], config: Config):
        self.results = results
        self.config = config

    def generate_comparison_report(self):
        logger.info("\n" + "="*70)
        logger.info(" MULTI-MODEL COMPARISON REPORT")
        logger.info("="*70)

        sorted_models = sorted(self.results.items(),
                             key=lambda x: x[1]['accuracy'],
                             reverse=True)

        print("\n{:<20} {:<12} {:<12} {:<12} {:<12}".format(
            "Model", "Accuracy", "Precision", "Recall", "F1-Score"
        ))
        print("-" * 70)

        for model_name, metrics in sorted_models:
            print("{:<20} {:<12.4f} {:<12.4f} {:<12.4f} {:<12.4f}".format(
                model_name,
                metrics['accuracy'],
                metrics['precision'],
                metrics['recall'],
                metrics['f1']
            ))

        logger.info(f"\n✓ Best Model: {sorted_models[0][0]} (Accuracy: {sorted_models[0][1]['accuracy']:.4f})")

        self._plot_model_comparison()

    def _plot_model_comparison(self):
        models = list(self.results.keys())
        metrics = ['accuracy', 'precision', 'recall', 'f1']

        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        axes = axes.flatten()

        for idx, metric in enumerate(metrics):
            values = [self.results[m][metric] for m in models]

            axes[idx].barh(models, values, color='skyblue', edgecolor='black')
            axes[idx].set_xlabel(metric.capitalize())
            axes[idx].set_title(f'Model Comparison - {metric.capitalize()}',
                              fontweight='bold')
            axes[idx].grid(axis='x', alpha=0.3)

            for i, v in enumerate(values):
                axes[idx].text(v + 0.005, i, f'{v:.4f}', va='center')

        plt.tight_layout()
        plt.show()

def load_dataset(config: Config) -> Tuple[List[Path], List[int], List[Path], List[int], List[str]]:
    train_path = Path(config.TRAIN_PATH)
    test_path = Path(config.TEST_PATH)

    if not train_path.exists() or not test_path.exists():
        raise FileNotFoundError(f"Dataset paths not found!")

    class_names = sorted([d.name for d in train_path.iterdir() if d.is_dir()])

    train_images, train_labels = [], []
    for label, class_name in enumerate(class_names):
        class_dir = train_path / class_name
        images = list(class_dir.glob('*.jpg')) + list(class_dir.glob('*.png'))
        train_images.extend(images)
        train_labels.extend([label] * len(images))

    test_images, test_labels = [], []
    for label, class_name in enumerate(class_names):
        class_dir = test_path / class_name
        images = list(class_dir.glob('*.jpg')) + list(class_dir.glob('*.png'))
        test_images.extend(images)
        test_labels.extend([label] * len(images))

    return train_images, train_labels, test_images, test_labels, class_names

def generate_visualizations(config: Config, train_images: List[Path],
                          train_labels: List[int], test_labels: List[int],
                          class_names: List[str], preprocessor: MRIPreprocessor):
    """Generate all preprocessing and augmentation visualizations"""
    logger.info("\n" + "="*70)
    logger.info("GENERATING VISUALIZATIONS")
    logger.info("="*70)

    visualizer = DataVisualizer(config)

    visualizer.plot_dataset_distribution(train_labels, test_labels, class_names)

    sample_images = []
    sample_labels = []

    for class_idx in range(len(class_names)):
        class_images = [img for img, lbl in zip(train_images, train_labels) if lbl == class_idx]
        if class_images:
            sample_idx = np.random.randint(0, min(len(class_images), 10))
            sample_path = class_images[sample_idx]
            img = cv2.imread(str(sample_path))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            sample_images.append(img)
            sample_labels.append(class_idx)

    sample_images = sample_images[:config.NUM_SAMPLE_IMAGES]
    sample_labels = sample_labels[:config.NUM_SAMPLE_IMAGES]

    visualizer.visualize_preprocessing_steps(
        sample_images, preprocessor, class_names, sample_labels
    )

    if sample_images:
        processed_img = preprocessor.process_pipeline(sample_images[0])
        processed_img = cv2.resize(processed_img, config.IMG_SIZE)
        visualizer.visualize_augmentation_examples(
            processed_img, class_names[sample_labels[0]]
        )

    logger.info("✓ All visualizations generated successfully")
    logger.info("="*70 + "\n")

def print_dataset_statistics(train_labels: List[int], test_labels: List[int],
                            class_names: List[str], config: Config):

    logger.info("\n DATASET STATISTICS")

    train_counts = Counter(train_labels)
    test_counts = Counter(test_labels)

    logger.info(f"\n{'Class':<20} {'Training':<15} {'Test':<15}")

    for idx, class_name in enumerate(class_names):
        train_count = train_counts[idx]
        test_count = test_counts[idx]
        logger.info(f"{class_name:<20} {train_count:<15} {test_count:<15}")

    logger.info(f"{'TOTAL':<20} {len(train_labels):<15} {len(test_labels):<15}")

    logger.info("\n AUGMENTATION ESTIMATES (per epoch)")

    base_train = len(train_labels)
    horizontal_flip_est = int(base_train * 0.5)
    rotation_est = int(base_train * 0.5)
    brightness_est = int(base_train * 0.5)
    contrast_est = int(base_train * 0.5)
    random_erasing_est = int(base_train * config.RANDOM_ERASING_PROB)

    logger.info(f"Base training images: {base_train}")
    logger.info(f"Horizontal flip augmentation: ~{horizontal_flip_est} images")
    logger.info(f"Rotation augmentation: ~{rotation_est} images")
    logger.info(f"Brightness augmentation: ~{brightness_est} images")
    logger.info(f"Contrast augmentation: ~{contrast_est} images")
    logger.info(f"Random erasing augmentation: ~{random_erasing_est} images")
    logger.info(f"\nNote: Multiple augmentations can be applied to the same image")
    logger.info("\n")

def main():
    config = Config()

    np.random.seed(config.RANDOM_STATE)
    torch.manual_seed(config.RANDOM_STATE)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(config.RANDOM_STATE)

    logger.info("\n STARTING MULTI-MODEL MRI CLASSIFICATION PIPELINE")

    train_images, train_labels, test_images, test_labels, class_names = load_dataset(config)

    print_dataset_statistics(train_labels, test_labels, class_names, config)

    preprocessor = MRIPreprocessor(config)

    if config.VISUALIZE_STEPS:
        generate_visualizations(config, train_images, train_labels,
                              test_labels, class_names, preprocessor)

    train_dataset = MRIDataset(train_images, train_labels, config,
                              preprocessor, mode='train')
    test_dataset = MRIDataset(test_images, test_labels, config,
                             preprocessor, mode='test')

    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE,
                            shuffle=True, num_workers=config.NUM_WORKERS,
                            pin_memory=config.PIN_MEMORY)

    test_loader = DataLoader(test_dataset, batch_size=config.BATCH_SIZE,
                           shuffle=False, num_workers=config.NUM_WORKERS,
                           pin_memory=config.PIN_MEMORY)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}\n")

    model_names = ['resnet-50', 'resnet-101', 'efficientnet-b0',
                  'mobilenet-v2', 'densenet-201']

    all_results = {}

    for model_name in model_names:
        logger.info(f"\n{'='*70}")
        logger.info(f"Training {model_name.upper()}")
        logger.info(f"{'='*70}")

        model = ModelFactory.create_model(model_name, len(class_names))

        trainer = ModelTrainer(model, model_name, config,
                             train_loader, class_names)
        history = trainer.train()

        evaluator = ModelEvaluator(model, model_name, test_loader,
                                  class_names, device)
        results = evaluator.evaluate()

        all_results[model_name] = results

    comparison = MultiModelComparison(all_results, config)
    comparison.generate_comparison_report()

    logger.info("\nPREPROCESSING SUMMARY")

    logger.info("\nPreprocessing operations applied:")
    for operation, count in preprocessor.preprocessing_stats.items():
        logger.info(f"  {operation.replace('_', ' ').title()}: {count} images")

    logger.info("\nALL PROCESSES COMPLETED!")


if __name__ == "__main__":
    main()
