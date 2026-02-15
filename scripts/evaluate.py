#!/usr/bin/env python3
"""
Evaluation script for DeepNet framework
Usage: python scripts/evaluate.py --dataset datasets/data_1 --checkpoint checkpoints/best.pth
"""

import argparse
import sys
import os
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / 'build'))

import deepnet_backend as backend
from deepnet.python.data import ImageFolderDataset, DataLoader, ensure_dataset_extracted
from deepnet.python.models import build_model_from_config, load_checkpoint
import time

def flatten_batch(images):
    flat = []
    extend = flat.extend
    for img in images:
        extend(img)
    return flat

def evaluate(model, dataloader, criterion, num_classes):
    """Evaluate model with per-class metrics"""
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    
    # Per-class accuracy
    class_correct = [0] * num_classes
    class_total = [0] * num_classes
    
    for images, labels in dataloader:
        # Convert to tensor
        batch_images = flatten_batch(images)
        
        batch_size = len(images)
        input_tensor = backend.Tensor.from_data(
            batch_images,
            [batch_size, 3, 32, 32],
            requires_grad=False
        )
        
        outputs = model(input_tensor)
        loss_tensor = criterion.forward(outputs, labels)
        loss = loss_tensor.data[0]
        total_loss += loss
        
        for i in range(batch_size):
            max_idx = max(range(num_classes), key=lambda j: outputs.data[i * num_classes + j])
            true_label = labels[i]
            
            if max_idx == true_label:
                correct += 1
                class_correct[true_label] += 1
            
            total += 1
            class_total[true_label] += 1
    
    avg_loss = total_loss / len(dataloader)
    accuracy = 100. * correct / total
    
    # Calculate per-class accuracies
    class_accuracies = {}
    for i in range(num_classes):
        if class_total[i] > 0:
            class_accuracies[i] = 100. * class_correct[i] / class_total[i]
        else:
            class_accuracies[i] = 0.0
    
    return avg_loss, accuracy, class_accuracies


def main():
    parser = argparse.ArgumentParser(description='Evaluate DeepNet CNN')
    parser.add_argument('--dataset', type=str, required=True, help='Path to dataset directory')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to checkpoint file')
    parser.add_argument('--config', type=str, default='configs/model_config.yaml', help='Path to model configuration YAML file')
    parser.add_argument('--batch-size', type=int, default=64)
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("DeepNet Evaluation")
    print("=" * 70)
    
    # Load dataset (use validation split)
    print(f"\nLoading dataset: {args.dataset}")
    dataset_start = time.time()
    
    dataset = ImageFolderDataset(args.dataset, image_size=32, train=False, val_split=0.2)
    
    dataset_load_time = time.time() - dataset_start
    print(f"Dataset loading time: {dataset_load_time:.2f} seconds")
    
    num_classes = len(dataset.classes)
    print(f"Number of classes: {num_classes}")
    print(f"Test samples: {len(dataset)}")
    
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
    
    # Build model
    print(f"\nBuilding model from: {args.config}")
    model, config = build_model_from_config(args.config, num_classes)
    
    # Load checkpoint
    print(f"Loading checkpoint: {args.checkpoint}")
    load_checkpoint(model, args.checkpoint)
    
    # Evaluate
    print("\n" + "=" * 70)
    print("Evaluating...")
    print("=" * 70)
    
    criterion = backend.CrossEntropyLoss()
    loss, accuracy, class_accuracies = evaluate(model, dataloader, criterion, num_classes)
    
    print(f"\nOverall Results:")
    print(f"  Loss: {loss:.4f}")
    print(f"  Accuracy: {accuracy:.2f}%")
    
    print(f"\nPer-Class Accuracy:")
    for class_idx, class_name in enumerate(dataset.classes):
        print(f"  {class_name}: {class_accuracies[class_idx]:.2f}%")
    
    print("\n" + "=" * 70)


if __name__ == '__main__':
    main()
