"""
Data loading utilities with train/val split support
Optimized: Preloads dataset into RAM for fast GPU training.
"""

import cv2
from pathlib import Path
import random
import zipfile


# ==========================================================
# Dataset Auto-Extraction
# ==========================================================

def ensure_dataset_extracted(dataset_name):
    datasets_dir = Path("datasets")
    zip_path = datasets_dir / f"{dataset_name}.zip"
    extract_path = datasets_dir / dataset_name

    if extract_path.exists():
        return str(extract_path)

    if zip_path.exists():
        print(f"Extracting {zip_path}...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(datasets_dir)
        print(f"Extracted to {extract_path}")
        return str(extract_path)

    raise FileNotFoundError(
        f"Neither {zip_path} nor {extract_path} found."
    )


# ==========================================================
# ImageFolderDataset (Preloaded Version)
# ==========================================================

class ImageFolderDataset:
    """
    Folder dataset with full RAM preload.
    All images are processed once in __init__.
    __getitem__ becomes O(1).
    """

    def __init__(self, root_dir,
                 image_size=32,
                 channels=3,
                 train=True,
                 val_split=0.2,
                 augmentation=None,
                 seed=42):

        self.root_dir = Path(root_dir)
        self.image_size = image_size
        self.channels = channels
        self.train = train
        self.val_split = val_split
        self.augmentation = augmentation or {}

        # --------------------------
        # Discover Classes
        # --------------------------
        self.classes = sorted(
            [d.name for d in self.root_dir.iterdir() if d.is_dir()]
        )
        self.class_to_idx = {
            cls: idx for idx, cls in enumerate(self.classes)
        }

        # --------------------------
        # Collect Image Paths
        # --------------------------
        all_samples = []

        for class_name in self.classes:
            class_dir = self.root_dir / class_name
            class_idx = self.class_to_idx[class_name]

            for img_path in class_dir.glob('*'):
                if img_path.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']:
                    all_samples.append((str(img_path), class_idx))

        # --------------------------
        # Split
        # --------------------------
        random.seed(seed)
        random.shuffle(all_samples)

        split_idx = int(len(all_samples) * (1 - val_split))

        if train:
            self.samples = all_samples[:split_idx]
        else:
            self.samples = all_samples[split_idx:]

        print(f"Preloading {len(self.samples)} "
              f"{'train' if train else 'val'} samples...")

        # --------------------------
        # Preload Into RAM
        # --------------------------
        self.data = []
        self.labels = []

        append_data = self.data.append
        append_label = self.labels.append

        for img_path, label in self.samples:

            # Load image
            if self.channels == 1:
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            else:
                img = cv2.imread(img_path)

            if img is None:
                raise ValueError(f"Failed to load image: {img_path}")

            # Resize
            img = cv2.resize(img, (self.image_size, self.image_size))

            # Convert BGR -> RGB
            if self.channels == 3:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            # Augmentation (only once during preload)
            if self.train and self.augmentation.get('enabled', False):
                img = self._augment(img)

            # Normalize (OpenCV uses NumPy internally — allowed)
            img = img.astype('float32')
            img /= 255.0

            # Convert HWC → CHW efficiently
            if self.channels == 1:
                img_list = img.flatten().tolist()
            else:
                ch = cv2.split(img)
                img_list = []
                img_list.extend(ch[0].flatten().tolist())
                img_list.extend(ch[1].flatten().tolist())
                img_list.extend(ch[2].flatten().tolist())

            append_data(img_list)
            append_label(label)

        print(f"Preload complete. "
              f"{len(self.data)} samples in RAM.")

    # --------------------------
    # Dataset Interface
    # --------------------------

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

    # --------------------------
    # Augmentation
    # --------------------------

    def _augment(self, img):

        # Horizontal flip
        if self.augmentation.get('flip', True) and random.random() > 0.5:
            img = cv2.flip(img, 1)

        # Rotation
        if self.augmentation.get('rotate', True):
            angle = random.uniform(-15, 15)
            h, w = img.shape[:2]
            M = cv2.getRotationMatrix2D((w/2, h/2), angle, 1.0)
            img = cv2.warpAffine(img, M, (w, h))

        # Brightness / Contrast
        if (self.augmentation.get('brightness', True) or
                self.augmentation.get('contrast', True)):

            alpha = random.uniform(0.8, 1.2) \
                if self.augmentation.get('contrast', True) else 1.0

            beta = random.uniform(-20, 20) \
                if self.augmentation.get('brightness', True) else 0

            img = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)

        return img


# ==========================================================
# DataLoader
# ==========================================================

class DataLoader:
    """
    Simple, lightweight DataLoader.
    Now very fast because dataset is preloaded.
    """

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

            append_img = images.append
            append_lbl = labels.append

            for idx in batch_indices:
                img, label = self.dataset[idx]
                append_img(img)
                append_lbl(label)

            yield images, labels
