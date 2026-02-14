"""Data loading utilities with train/val split support"""

import os
import cv2
from pathlib import Path
import random
import zipfile


def ensure_dataset_extracted(dataset_name):
    """
    Auto-extract dataset if zip exists but folder doesn't.
    
    Args:
        dataset_name: 'data_1' or 'data_2'
    
    Returns:
        Path to extracted dataset folder
    """
    datasets_dir = Path("datasets")
    zip_path = datasets_dir / f"{dataset_name}.zip"
    extract_path = datasets_dir / dataset_name
    
    # If folder exists, return it
    if extract_path.exists():
        return str(extract_path)
    
    # If zip exists, extract it
    if zip_path.exists():
        print(f"Extracting {zip_path}...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(datasets_dir)
        print(f"Extracted to {extract_path}")
        return str(extract_path)
    
    # Neither exists - error
    raise FileNotFoundError(
        f"Neither {zip_path} nor {extract_path} found. "
        f"Please place {dataset_name}.zip in datasets/ folder."
    )


class ImageFolderDataset:
    """
    Load images from folder structure where each subdirectory is a class.
    
    Directory structure:
        datasets/data_1/
            class_0/
                img1.png
                img2.png
            class_1/
                img1.png
    
    Automatically splits into train/val sets.
    """
    
    def __init__(self, root_dir, image_size=32, channels=3, train=True, val_split=0.2, 
                 augmentation=None, seed=42):
        """
        Args:
            root_dir: Root directory containing class folders
            image_size: Resize images to this size
            channels: Number of color channels (1=grayscale, 3=RGB)
            train: If True, load train split; if False, load val split
            val_split: Fraction of data to use for validation
            augmentation: Dict with augmentation settings
            seed: Random seed for reproducible splits
        """
        self.root_dir = Path(root_dir)
        self.image_size = image_size
        self.channels = channels
        self.train = train
        self.val_split = val_split
        self.augmentation = augmentation or {}
        
        # Discover classes
        self.classes = sorted([d.name for d in self.root_dir.iterdir() if d.is_dir()])
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        
        # Load all image paths and labels
        all_samples = []
        for class_name in self.classes:
            class_dir = self.root_dir / class_name
            class_idx = self.class_to_idx[class_name]
            
            for img_path in class_dir.glob('*'):
                if img_path.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']:
                    all_samples.append((str(img_path), class_idx))
        
        # Shuffle and split
        random.seed(seed)
        random.shuffle(all_samples)
        
        split_idx = int(len(all_samples) * (1 - val_split))
        
        if train:
            self.samples = all_samples[:split_idx]
        else:
            self.samples = all_samples[split_idx:]
        
        print(f"Loaded {len(self.samples)} {'train' if train else 'val'} samples "
              f"from {len(self.classes)} classes")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        
        # Load image using OpenCV
        if self.channels == 1:
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        else:
            img = cv2.imread(img_path)
        if img is None:
            raise ValueError(f"Failed to load image: {img_path}")
        
        # Resize
        img = cv2.resize(img, (self.image_size, self.image_size))
        
        # Convert to RGB if color
        if self.channels == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Apply augmentation if training
        if self.train and self.augmentation.get('enabled', False):
            img = self._augment(img)
        
        # Convert to CHW format and normalize to [0, 1]
        h, w = self.image_size, self.image_size
        img_list = []
        
        if self.channels == 1:
            # Grayscale: single channel
            for i in range(h):
                for j in range(w):
                    img_list.append(float(img[i, j]) / 255.0)
        else:
            # RGB: 3 channels
            for c in range(3):
                for i in range(h):
                    for j in range(w):
                        img_list.append(float(img[i, j, c]) / 255.0)
        
        return img_list, label
    
    def _augment(self, img):
        """Apply data augmentation - OpenCV only"""
        # Random horizontal flip
        if self.augmentation.get('flip', True) and random.random() > 0.5:
            img = cv2.flip(img, 1)
        
        # Random rotation
        if self.augmentation.get('rotate', True):
            angle = random.uniform(-15, 15)
            h, w = img.shape[:2]
            M = cv2.getRotationMatrix2D((w/2, h/2), angle, 1.0)
            img = cv2.warpAffine(img, M, (w, h))
        
        # Random brightness and contrast using OpenCV
        if self.augmentation.get('brightness', True) or self.augmentation.get('contrast', True):
            alpha = random.uniform(0.8, 1.2) if self.augmentation.get('contrast', True) else 1.0
            beta = random.uniform(-20, 20) if self.augmentation.get('brightness', True) else 0
            img = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)
        
        return img


class DataLoader:
    """Simple DataLoader for batching"""
    
    def __init__(self, dataset, batch_size=32, shuffle=True):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_samples = len(dataset)
    
    def __len__(self):
        return (self.num_samples + self.batch_size - 1) // self.batch_size
    
    def __iter__(self):
        indices = list(range(self.num_samples))
        if self.shuffle:
            random.shuffle(indices)
        
        for i in range(0, self.num_samples, self.batch_size):
            batch_indices = indices[i:i + self.batch_size]
            
            images = []
            labels = []
            
            for idx in batch_indices:
                img, label = self.dataset[idx]  # img is flat list[float]
                images.append(img)
                labels.append(label)
            
            # images: list of lists (batch_size x 3072)
            # labels: list of ints (batch_size)
            
            yield images, labels
