"""Visualization utilities for training/evaluation results"""

import os
from pathlib import Path


def save_training_log(log_path, epoch, train_loss, train_acc, val_loss=None, val_acc=None, epoch_time=None):
    """Save training metrics to CSV file"""
    
    # Create log directory if needed
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    
    # Check if file exists
    file_exists = os.path.isfile(log_path)
    
    with open(log_path, 'a') as f:
        # Write header if new file
        if not file_exists:
            f.write("epoch,train_loss,train_acc,val_loss,val_acc,epoch_time\n")
        
        # Write data
        f.write(f"{epoch},{train_loss:.6f},{train_acc:.4f},")
        
        if val_loss is not None:
            f.write(f"{val_loss:.6f},")
        else:
            f.write(",")
        
        if val_acc is not None:
            f.write(f"{val_acc:.4f},")
        else:
            f.write(",")
        
        if epoch_time is not None:
            f.write(f"{epoch_time:.2f}")
        
        f.write("\n")


def print_training_header():
    """Print formatted training header"""
    print("\n" + "="*80)
    print(f"{'Epoch':>6} | {'Train Loss':>12} | {'Train Acc':>10} | {'Time (s)':>10}")
    print("="*80)


def print_epoch_stats(epoch, max_epochs, train_loss, train_acc, epoch_time, val_loss=None, val_acc=None):
    """Print formatted epoch statistics"""
    print(f"{epoch:>6}/{max_epochs:<6} | {train_loss:>12.6f} | {train_acc:>9.2f}% | {epoch_time:>10.2f}")
    
    if val_loss is not None and val_acc is not None:
        print(f"{'':>6} | Val Loss: {val_loss:>10.6f} | Val Acc: {val_acc:>8.2f}%")


def print_confusion_matrix(confusion_matrix, class_names):
    """Print confusion matrix"""
    num_classes = len(class_names)
    
    print("\nConfusion Matrix:")
    print("="*60)
    
    # Print header
    print(f"{'True/Pred':<15}", end="")
    for name in class_names:
        print(f"{name[:8]:>10}", end="")
    print()
    
    print("-"*60)
    
    # Print matrix
    for i, true_class in enumerate(class_names):
        print(f"{true_class[:15]:<15}", end="")
        for j in range(num_classes):
            print(f"{confusion_matrix[i][j]:>10}", end="")
        print()
    
    print("="*60)


def plot_training_curves(log_path, output_path=None):
    """
    Plot training curves from log file
    Note: This requires matplotlib which is not in our allowed dependencies.
    For the assignment, we'll just document this as a future enhancement.
    """
    print(f"Training log saved to: {log_path}")
    print("Tip: You can visualize training curves using spreadsheet software or matplotlib")
