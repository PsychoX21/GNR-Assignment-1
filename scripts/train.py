#!/usr/bin/env python3
"""
Training script for DeepNet framework
Usage: python scripts/train.py --dataset datasets/data_1 --config configs/model_config.yaml --epochs 50
"""

import argparse
import time
import sys
import os
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / 'build'))

import deepnet_backend as backend
from deepnet.python.data import ImageFolderDataset, DataLoader, ensure_dataset_extracted
from deepnet.python.models import build_model_from_config, calculate_model_stats, save_checkpoint


def train_epoch(model, dataloader, criterion, optimizer, epoch, use_cuda=False):
    """Train for one epoch"""
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    
    for batch_idx, (images, labels) in enumerate(dataloader):
        optimizer.zero_grad()
        
        # Convert list[list[float]] to flat list for backend
        batch_images = []
        for img in images:  # Each img is list of 3072 floats (3*32*32)
            batch_images.extend(img)
        
        # Create tensor: [batch_size, 3, 32, 32]
        batch_size = len(images)
        input_tensor = backend.Tensor.from_data(
            batch_images,
            [batch_size, 3, 32, 32],
            requires_grad=False,
            cuda=use_cuda
        )
        
        # Forward pass
        outputs = model(input_tensor)
        
        # Compute loss (also computes input_grad internally)
        loss_tensor = criterion.forward(outputs, labels)
        loss = loss_tensor.data[0]
        
        # Backward pass: propagate gradients from loss through the network
        # Get dL/d(logits) from the criterion and backprop through all layers
        input_grad = criterion.get_input_grad()
        if input_grad is not None:
            model.backward(input_grad)
        
        # Update weights using computed gradients
        optimizer.step()
        
        # Calculate accuracy
        num_classes = outputs.shape[1]
        for i in range(batch_size):
            max_idx = max(range(num_classes), key=lambda j: outputs.data[i * num_classes + j])
            if max_idx == labels[i]:
                correct += 1
            total += 1
        
        total_loss += loss
        
        if batch_idx % 10 == 0:
            print(f'Epoch [{epoch}] Batch [{batch_idx}/{len(dataloader)}] '
                  f'Loss: {loss:.4f} Acc: {100.*correct/total:.2f}%')
    
    avg_loss = total_loss / len(dataloader)
    accuracy = 100. * correct / total
    
    return avg_loss, accuracy


def evaluate(model, dataloader, criterion, use_cuda=False):
    """Evaluate model"""
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    
    for images, labels in dataloader:
        # Convert to tensor
        batch_images = []
        for img in images:
            batch_images.extend(img)
        
        batch_size = len(images)
        input_tensor = backend.Tensor.from_data(
            batch_images,
            [batch_size, 3, 32, 32],
            requires_grad=False,
            cuda=use_cuda
        )
        
        outputs = model(input_tensor)
        loss_tensor = criterion.forward(outputs, labels)
        loss = loss_tensor.data[0]
        total_loss += loss
        
        num_classes = outputs.shape[1]
        for i in range(batch_size):
            max_idx = max(range(num_classes), key=lambda j: outputs.data[i * num_classes + j])
            if max_idx == labels[i]:
                correct += 1
            total += 1
    
    avg_loss = total_loss / len(dataloader)
    accuracy = 100. * correct / total
    
    return avg_loss, accuracy


def main():
    parser = argparse.ArgumentParser(description='Train DeepNet CNN')
    parser.add_argument('--dataset', type=str, required=True, help='Path to dataset directory')
    parser.add_argument('--config', type=str, required=True, help='Path to model configuration YAML file')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=64, help='Batch size')
    parser.add_argument('--val-split', type=float, default=0.2, help='Validation split ratio')
    parser.add_argument('--checkpoint-dir', type=str, default='checkpoints', help='Directory to save checkpoints')
    
    args = parser.parse_args()
    
    # Create checkpoint directory
    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    os.makedirs('logs', exist_ok=True)
    
    print("=" * 70)
    print("DeepNet Training")
    print("=" * 70)
    
    # Load datasets with train/val split
    print(f"\nLoading dataset: {args.dataset}")
    dataset_start = time.time()
    
    train_dataset = ImageFolderDataset(args.dataset, image_size=32, train=True, 
                                       val_split=args.val_split, 
                                       augmentation={'enabled': True})
    val_dataset = ImageFolderDataset(args.dataset, image_size=32, train=False, 
                                     val_split=args.val_split)
    
    dataset_load_time = time.time() - dataset_start
    print(f"Dataset loading time: {dataset_load_time:.2f} seconds")
    
    num_classes = len(train_dataset.classes)
    print(f"Number of classes: {num_classes}")
    print(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    
    # Build model from config
    print(f"\nBuilding model from: {args.config}")
    model, config = build_model_from_config(args.config, num_classes)
    
    stats = calculate_model_stats(model, [args.batch_size, 3, 32, 32])
    print(f"\nModel Statistics:")
    print(f"  Parameters: {stats['parameters']:,}")
    print(f"  MACs: {stats['macs']:,}")
    print(f"  FLOPs: {stats['flops']:,}")
    
    # Create loss and optimizer
    criterion = backend.CrossEntropyLoss()
    params = model.parameters()
    
    training_config = config['training']
    optimizer_type = training_config.get('optimizer', 'Adam')
    
    if optimizer_type == 'SGD':
        optimizer = backend.SGD(params, lr=training_config.get('learning_rate', 0.01),
                                momentum=training_config.get('momentum', 0.9),
                                weight_decay=training_config.get('weight_decay', 0.0001))
    else:
        optimizer = backend.Adam(params, lr=training_config.get('learning_rate', 0.001))
    
    print(f"\nOptimizer: {optimizer_type}, LR: {training_config.get('learning_rate')}")
    
    # Check CUDA availability
    use_cuda = False
    try:
        use_cuda = backend.is_cuda_available()
    except:
        use_cuda = False
    
    cuda_status = "enabled" if use_cuda else "disabled (CPU only)"
    print(f"\nCUDA status: {cuda_status}")
    
    # Training loop
    print("\n" + "=" * 70)
    print("Training Started")
    print("=" * 70)
    
    best_val_acc = 0.0
    
    for epoch in range(1, args.epochs + 1):
        start = time.time()
        
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, epoch, use_cuda)
        val_loss, val_acc = evaluate(model, val_loader, criterion, use_cuda)
        
        epoch_time = time.time() - start
        
        print(f"\n{'='*70}")
        print(f"Epoch {epoch}/{args.epochs}")
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
        print(f"Time: {epoch_time:.2f}s")
        print(f"{'='*70}\n")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            save_checkpoint(model, optimizer, epoch, val_loss, 'checkpoints/best.pth')
            print(f"[BEST] Saved best model (Val Acc: {best_val_acc:.2f}%)")
        
        # Save periodic checkpoints
        if epoch % 10 == 0:
            save_checkpoint(model, optimizer, epoch, val_loss, f'checkpoints/epoch_{epoch}.pth')
    
    print(f"\n{'='*70}")
    print("Training Complete!")
    print(f"Best Validation Accuracy: {best_val_acc:.2f}%")
    print(f"{'='*70}\n")


if __name__ == '__main__':
    main()
