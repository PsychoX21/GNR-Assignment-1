# DeepNet: Custom Deep Learning Framework

**GNR638 Machine Learning for Remote Sensing — Assignment 1**

A high-performance CNN framework built from scratch with a C++ backend and Python frontend. Implements all tensor operations, layers, optimizers, and training utilities without any external ML libraries. Includes OpenMP for multi-threaded CPU parallelization and optional CUDA for GPU acceleration.

---

## Quick Start

### Prerequisites

| Requirement | Minimum | Notes |
|---|---|---|
| Python | 3.10+ | 3.12+ recommended |
| CMake | 3.15+ | [cmake.org/download](https://cmake.org/download/) |
| Git | Any | For cloning pybind11 |
| C++ Compiler | C++17 support | See platform-specific below |
| OpenMP | Optional | Auto-detected, speeds up CPU |
| CUDA Toolkit | Optional (11.0+) | Auto-detected, enables GPU |

**Platform-specific compilers:**
- **Windows**: Visual Studio 2019+ with "Desktop development with C++" workload
- **Linux**: `sudo apt install build-essential cmake python3-dev`
- **macOS**: Xcode Command Line Tools (`xcode-select --install`)

### Build & Run

```bash
# 1. Setup environment (first time only)
make setup

# 2. Activate virtual environment
# Linux/macOS:
source venv/bin/activate
# Windows PowerShell:
.\venv\Scripts\activate
# Windows CMD:
venv\Scripts\activate

# 3. Build C++ backend and install
make build install

# 4. Verify everything works
make test

# 5. Train
make train DATA=data_1 CONFIG=configs/mnist_config.yaml EPOCHS=30
make train DATA=data_2 CONFIG=configs/cifar100_config.yaml EPOCHS=50
```

**One-command build** (after activating venv):
```bash
make    # Does: setup + build + install
```

---

## Makefile Targets

| Target | Description |
|---|---|
| `make` | Full setup + build + install |
| `make setup` | Create venv, install deps, clone pybind11 |
| `make build` | Compile C++ backend with CMake |
| `make install` | Copy compiled module and install Python package |
| `make test` | Run all tests (layers + gradient + CUDA) |
| `make test-layers` | Run layer/tensor operation tests |
| `make test-gradient` | Run gradient and training convergence test |
| `make test-cuda` | Run CUDA GPU acceleration tests |
| `make train` | Train model (`DATA=`, `CONFIG=`, `EPOCHS=`, `BATCH_SIZE=`) |
| `make eval` | Evaluate model (`DATA=`, `MODEL=`) |
| `make clean` | Remove build artifacts |
| `make distclean` | Deep clean (also removes venv, pybind11) |

**Examples:**
```bash
# MNIST (10 digits)
make train DATA=data_1 CONFIG=configs/mnist_config.yaml EPOCHS=30
make eval DATA=data_1 MODEL=checkpoints/best_data_1.pth

# CIFAR-100 (100 classes)
make train DATA=data_2 CONFIG=configs/cifar100_config.yaml EPOCHS=50
make eval DATA=data_2 MODEL=checkpoints/best_data_2.pth CONFIG=configs/cifar100_config.yaml

make test-cuda  # Verify GPU acceleration
```

---

## Project Structure

```
GNR-Assignment-1/
├── Makefile                        # Cross-platform build system
├── CMakeLists.txt                  # CMake configuration
├── setup.py                        # Python package installer
├── requirements.txt                # Python dependencies (opencv, pyyaml, tqdm)
│
├── deepnet/                        # Framework
│   ├── cpp/                        # C++ Backend
│   │   ├── include/
│   │   │   ├── tensor.hpp                 # Tensor with autograd support
│   │   │   ├── loss.hpp                   # CrossEntropyLoss, MSELoss
│   │   │   ├── layers/
│   │   │   │   ├── layer.hpp              # Conv2D, Linear, activations
│   │   │   │   ├── pooling.hpp            # MaxPool2D, AvgPool2D
│   │   │   │   └── batchnorm.hpp          # BatchNorm2D, BatchNorm1D, Dropout
│   │   │   ├── optimizers/
│   │   │   │   ├── optimizer.hpp          # SGD (with momentum), Adam
│   │   │   │   └── scheduler.hpp          # StepLR, CosineAnnealingLR
│   │   │   └── cuda/
│   │   │       ├── cuda_ops.hpp           # CUDA kernel declarations
│   │   │       └── cuda_utils.hpp         # CUDA utilities and fallbacks
│   │   └── src/
│   │       ├── tensor.cpp                 # Tensor ops with OpenMP + CUDA dispatch
│   │       ├── loss.cpp                   # Loss function implementations
│   │       ├── layers/
│   │       │   ├── layer.cpp              # Conv2D, Linear, activations (OpenMP)
│   │       │   ├── pooling.cpp            # Pooling layers (OpenMP)
│   │       │   └── batchnorm.cpp          # BatchNorm layers (OpenMP)
│   │       ├── optimizers/optimizer.cpp   # Optimizer implementations
│   │       └── cuda/cuda_kernels.cu       # CUDA GPU kernels
│   │
│   ├── bindings/bindings.cpp       # pybind11 Python bindings
│   └── python/
│       ├── data.py                 # Dataset loading, augmentation
│       ├── module.py               # PyTorch-like Module, Sequential
│       └── models.py               # YAML model builder, stats calculator
│
├── scripts/
│   ├── train.py                    # Training script
│   ├── evaluate.py                 # Evaluation script
│   ├── test_all_layers.py          # Layer correctness tests
│   ├── test_gradient.py            # Gradient and training tests
│   └── test_cuda.py                # CUDA GPU tests
│
├── configs/
│   ├── mnist_config.yaml           # Optimized for MNIST/data_1 (10 classes)
│   ├── cifar100_config.yaml        # Optimized for CIFAR-100/data_2 (100 classes)
│   ├── model_config.yaml           # Heavy CNN (for reference)
│   ├── model_config_fast.yaml      # Medium CNN (for reference)
│   └── model_config_simple.yaml    # Minimal LeNet
│
├── utils/
│   ├── metrics.py                  # Parameters, MACs, FLOPs calculation
│   └── visualization.py            # Logging utilities
│
└── datasets/                       # Data (auto-extracted from .zip)
    ├── data_1.zip                  # 10-class dataset
    └── data_2.zip                  # 100-class dataset
```

---

## Training

```bash
# MNIST (data_1) — ~5-10 min/epoch, target 97%+
python scripts/train.py --dataset datasets/data_1 --config configs/mnist_config.yaml --epochs 30 --batch-size 128

# CIFAR-100 (data_2) — ~10-15 min/epoch, target 35-45%
python scripts/train.py --dataset datasets/data_2 --config configs/cifar100_config.yaml --epochs 50 --batch-size 64
```

| Argument | Default | Description |
|---|---|---|
| `--dataset` | (required) | Path to dataset directory |
| `--config` | (required) | Path to model config YAML |
| `--epochs` | 50 | Number of training epochs |
| `--batch-size` | 64 | Batch size |
| `--val-split` | 0.2 | Train/validation split ratio |
| `--checkpoint-dir` | `checkpoints` | Where to save model checkpoints |

**Outputs:**
- `checkpoints/best_data_1.pth` — Best validation accuracy model (per dataset)
- `checkpoints/data_1_epoch_10.pth` — Periodic checkpoints (every 10 epochs)
- Console: per-epoch train/val loss, accuracy, timing

**CUDA:** If an NVIDIA GPU is detected, tensor operations automatically dispatch to GPU kernels. No code changes needed — the script prints `CUDA status: enabled` at startup.

---

## Evaluation

```bash
python scripts/evaluate.py --dataset datasets/data_1 --checkpoint checkpoints/best_data_1.pth
python scripts/evaluate.py --dataset datasets/data_2 --checkpoint checkpoints/best_data_2.pth --config configs/cifar100_config.yaml
```

Prints overall accuracy, loss, and per-class accuracy breakdown.

---

## Model Configuration

Models are defined in YAML config files. Two optimized configs are provided:

| Config | Architecture | Best For | Target Accuracy |
|---|---|---|---|
| `mnist_config.yaml` | LeNet (16→32), 2 conv blocks | data_1 (MNIST, 10 classes) | 97%+ |
| `cifar100_config.yaml` | 3-block CNN (32→64→128) | data_2 (CIFAR-100, 100 classes) | 35-45% |

The final layer uses `out_features: "num_classes"` which is automatically replaced based on the dataset.

### Config Format

```yaml
model:
  architecture:
    - type: "Conv2D"
      in_channels: 3
      out_channels: 32
      kernel_size: 3
      padding: 1
    - type: "BatchNorm2D"
      num_features: 32
    - type: "ReLU"
    - type: "MaxPool2D"
      kernel_size: 2
    # ... more layers
    - type: "Linear"
      in_features: 128
      out_features: "num_classes"  # Auto-replaced

training:
  optimizer: "Adam"       # Adam or SGD
  learning_rate: 0.001
  weight_decay: 0.0001
  scheduler:
    type: "StepLR"
    step_size: 15
    gamma: 0.5
```

---

## Dataset Format

| Dataset | Contents | Classes | Image Size |
|---|---|---|---|
| `data_1` | MNIST handwritten digits | 10 | 28×28 (loaded as 32×32 RGB) |
| `data_2` | CIFAR-100 natural images | 100 | 32×32 RGB |

Place `.zip` files in `datasets/`. They are auto-extracted on first training run.

```
datasets/data_1/
├── 0/
│   ├── img001.png
│   └── ...
├── 1/
└── ...
```

---

## OpenMP & CUDA Acceleration

### OpenMP (CPU Parallelization)
- **Auto-detected** at build time
- Parallelizes ~25 compute-heavy loops across all layers
- Falls back to single-threaded if not available
- Supported on Windows (MSVC), Linux (GCC), macOS (Clang with libomp)

### CUDA (GPU Acceleration)
- **Auto-detected** at build time (requires NVIDIA GPU + CUDA Toolkit)
- 6 CUDA-accelerated operations: add, mul, matmul, relu, sigmoid, tanh
- Uses copy-in → GPU compute → copy-out pattern
- Runtime check: `backend.is_cuda_available()` — falls back to CPU if no GPU
- Verify with: `make test-cuda`

**Build output** indicates what was detected:
```
-- OpenMP found - parallel CPU enabled
-- CUDA found - GPU acceleration enabled
```

---

## Implementation Details

### C++ Backend
- All tensor operations in C++17, compiled with optimization flags
- MSVC: `/O2 /arch:AVX2` | GCC/Clang: `-O3 -march=native -ffast-math`
- OpenMP parallelization across all compute-heavy loops
- Optional CUDA GPU kernels (tiled shared-memory matmul)

### Layers Implemented
| Layer | Forward | Backward |
|---|---|---|
| Conv2D | ✅ | ✅ |
| Linear | ✅ | ✅ |
| ReLU, LeakyReLU, Tanh, Sigmoid | ✅ | ✅ |
| MaxPool2D, AvgPool2D | ✅ | ✅ |
| BatchNorm2D, BatchNorm1D | ✅ | ✅ |
| Dropout | ✅ | ✅ |
| Flatten | ✅ | ✅ |

### Optimizers
- **SGD** with momentum and weight decay
- **Adam** with bias correction

### No External ML Libraries
- All computations implemented from scratch in C++
- No NumPy, PyTorch, TensorFlow, or JAX
- Only external libs: OpenCV (image I/O), pybind11 (bindings), PyYAML (config), tqdm (progress bars)

---

## Cross-Platform Compatibility

| Feature | Windows | Linux | macOS |
|---|---|---|---|
| Build system | ✅ MSVC | ✅ GCC | ✅ Clang |
| OpenMP | ✅ Built-in | ✅ Built-in | ✅ via `brew install libomp` |
| CUDA | ✅ | ✅ | ❌ (no NVIDIA GPUs) |
| Makefile | ✅ Auto-detects OS | ✅ | ✅ |
| CMake | ✅ Visual Studio generator | ✅ Unix Makefiles | ✅ Unix Makefiles |

The Makefile auto-detects the operating system and uses the appropriate commands (e.g., `copy` vs `cp`, `mkdir` vs `mkdir -p`).

---

## Troubleshooting

### "Module 'deepnet_backend' not found"
```bash
make clean && make build install
```

### "pybind11 not found"
```bash
git clone https://github.com/pybind/pybind11.git
```

### Build fails on Windows
- Install Visual Studio 2019+ with "Desktop development with C++" workload
- Ensure x64 architecture is targeted (64-bit)

### Build fails on Linux
```bash
sudo apt install build-essential cmake python3-dev
```

### Build fails on macOS
```bash
xcode-select --install
# For OpenMP:
brew install libomp
```

### CUDA-related build errors
- Ensure CUDA Toolkit is installed and `nvcc` is in PATH
- The build automatically falls back to CPU-only if CUDA is not found

### IDE Errors (clangd / IntelliSense)
CUDA/template-heavy C++ code may trigger IDE squiggles. These are IDE-only and **do not affect compilation**. The `make build` will succeed regardless.

---

## Assignment Compliance

- ✅ Custom framework from scratch (no PyTorch/TensorFlow/NumPy)
- ✅ C++ backend with Python bindings (+20 bonus)
- ✅ All required layers: Conv2D, Pooling, Linear, Activations, BatchNorm, Dropout
- ✅ Manual forward and backward passes for all layers
- ✅ SGD and Adam optimizers
- ✅ Dataset loading with measured time
- ✅ Parameters, MACs, FLOPs calculation
- ✅ OpenMP CPU parallelization
- ✅ CUDA GPU acceleration (optional)
- ✅ Training under 3 hours per dataset
- ✅ Only allowed libraries used

---

## License

Educational project for GNR638 coursework.
