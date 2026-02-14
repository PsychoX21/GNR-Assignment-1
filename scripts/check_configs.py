import sys
import os
sys.path.append(os.getcwd())
try:
    from deepnet.python.models import build_model_from_config, calculate_model_stats
except ImportError:
    print("Error importing deepnet. Make sure you are in project root.")
    sys.exit(1)

configs = [
    ('MNIST Baseline', 'configs/mnist_config.yaml', 10, [1, 28, 28], 128),
    ('CIFAR Fast', 'configs/cifar100_strided_fast.yaml', 100, [3, 32, 32], 256),
    ('CIFAR Balanced', 'configs/cifar100_balanced.yaml', 100, [3, 32, 32], 128)
]

print(f"{'Model':<20} | {'Batch':<5} | {'Params':<10} | {'MACs (Total)':<15} | {'MACs/Sample':<12}")
print("-" * 75)

for name, path, classes, shape, batch in configs:
    if os.path.exists(path):
        try:
            m, _ = build_model_from_config(path, classes)
            # Calculate stats for the specific batch size used in config
            full_shape = [batch] + shape
            s = calculate_model_stats(m, full_shape)
            
            params = s['parameters']
            macs = s['macs']
            macs_per_sample = macs / batch
            
            print(f"{name:<20} | {batch:<5} | {params:<10,} | {macs:<15,} | {int(macs_per_sample):<12,}")
        except Exception as e:
            print(f"{name:<20} | Error: {e}")
    else:
        print(f"{name:<20} | File not found")
