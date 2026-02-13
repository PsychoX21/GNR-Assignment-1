# DeepNet: Custom Deep Learning Framework

**GNR638 Machine Learning for Remote Sensing Assignment**

A high-performance CNN framework with C++ backend and Python 3.12 frontend, implementing all components from scratch.

---

## 🚀 Quick Start - Build Instructions

### For Graders and First-Time Users:

**Step 1: Setup Environment (One-Time)**
```bash
# Run this once to create venv and install dependencies
make setup

# Activate the virtual environment:
# On Linux/macOS:
source venv/bin/activate
# On Windows (PowerShell):
.\venv\Scripts\activate
# On Windows (CMD):
venv\Scripts\activate
```

**Step 2: Build and Install (Required)**
```bash
# Build C++ backend and install Python package
make build
make install

# OR run both at once:
make build install
```

**Step 3: Train and Evaluate**
```bash
# Train on 10-class dataset (quick test with 2 epochs)
python scripts/train.py --dataset datasets/data_1 --config configs/model_config.yaml --epochs 2

# Full training (50 epochs)
python scripts/train.py --dataset datasets/data_1 --config configs/model_config.yaml --epochs 50

# Evaluate
python scripts/evaluate.py --dataset datasets/data_1 --checkpoint checkpoints/best.pth
```

### Simplified One-Command Build:

If `make setup` completes successfully, you can run everything at once:
```bash
make              # Does: setup + build + install
source venv/bin/activate  # Linux/macOS
python scripts/train.py --dataset datasets/data_1 --config configs/model_config.yaml --epochs 2
```

### Using Makefile Shortcuts:

```bash
# Train using Makefile (after build + install)
make train DATA=data_1 EPOCHS=50

# Evaluate using Makefile
make eval DATA=data_1 MODEL=checkpoints/best.pth
```

---

## Project Structure

```
Assignment 1/
├── Makefile                    # Build system
├── CMakeLists.txt              # CMake configuration
├── setup.py                    # Python package installer
├── requirements.txt            # Python dependencies
│
├── deepnet/                    # Custom Framework
│   ├── cpp/                    # C++ Backend
│   │   ├── include/            # Headers
│   │   │   ├── tensor.hpp              # Tensor with autograd
│   │   │   ├── loss.hpp                # Loss functions
│   │   │   ├── layers/
│   │   │   │   ├── layer.hpp           # Conv2D, Linear, activations
│   │   │   │   ├── pooling.hpp         # Pooling layers
│   │   │   │   └── batchnorm.hpp       # BatchNorm, Dropout
│   │   │   ├── optimizers/
│   │   │   │   ├── optimizer.hpp       # SGD, Adam
│   │   │   │   └── scheduler.hpp       # LR schedulers
│   │   │   └── cuda/                    # CUDA headers (optional GPU)
│   │   └── src/                # Implementations
│   ├── bindings/bindings.cpp   # pybind11 Python bindings
│   └── python/                 # Python API
│       ├── data.py             # Dataset loader
│       ├── module.py           # Module system
│       └── models.py           # Model builder
│
├── scripts/
│   ├── train.py                # Training script
│   └── evaluate.py             # Evaluation script
│
├── configs/
│   └── model_config.yaml       # CNN model configuration
│
├── utils/
│   ├── metrics.py              # Parameters, MACs, FLOPs
│   └── visualization.py        # Logging utilities
│
└── datasets/                   # Data folder
    ├── data_1.zip              # 10-class dataset
    ├── data_2.zip              # 100-class dataset
    ├── data_1/                 # Auto-extracted
    └── data_2/                 # Auto-extracted
```

---

## Build System

### Prerequisites

**System:**
- Python 3.12+
- CMake 3.12+
- Git

**Compiler:**
- **Windows**: Visual Studio 2019+ (MSVC) or MinGW-w64
- **Linux/Mac**: GCC 9+ or Clang 10+

**Optional:**
- CUDA Toolkit 11.0+ (for GPU acceleration)

**Important:** The framework is built for 64-bit systems. Ensure your compiler targets x64/x86_64 architecture. Using 32-bit compilers will cause build failures.

### What is setup.py?

`setup.py` is a Python packaging file that:
1. Makes `deepnet` importable as a Python package
2. Links the compiled C++ backend (`deepnet_backend.so` / `.pyd`) to Python
3. Installs dependencies (opencv-python, pyyaml, tqdm)
4. Enables `import deepnet_backend` in your scripts

It's required to connect the C++ code to Python.

### Build Process

The Makefile **automatically detects your OS** (Windows/Linux/macOS) and uses the correct commands.

```bash
# Option 1: All-in-one (does setup + build + install)
make

# Option 2: Step-by-step (recommended for understanding)
make setup      # Clone pybind11, create venv, install Python deps

# Activate virtual environment:
source venv/bin/activate      # Linux/macOS
.\venv\Scripts\activate       # Windows PowerShell
venv\Scripts\activate         # Windows CMD

make build      # Compile C++ backend
make install    # Install Python package
```

**What happens during build:**
1. CMake detects your compiler and CUDA (if available)
2. C++ code compiled with optimization (`-O3 -march=native` for GCC/Clang, `/O2` for MSVC)
3. Creates `build/deepnet_backend.so` (Linux/Mac) or `.pyd` (Windows)
4. `make install` copies it to project root and links it to Python

**Cross-Platform Support:**
- ✅ **Windows**: Auto-detects and uses appropriate commands (mkdir, copy, etc.)
- ✅ **Linux/WSL**: Uses bash/sh commands
- ✅ **macOS**: Fully supported

---

## Dataset Format

Datasets are auto-extracted on first use:

```
datasets/
├── data_1.zip          # Put zip files here
├── data_2.zip
├── data_1/             # Auto-extracted on first train
│   ├── class_0/
│   │   ├── img1.png
│   │   └── ...
│   ├── class_1/
│   └── ...
└── data_2/             # Auto-extracted on first train
    ├── class_0/
    └── ...
```

The framework automatically:
- Detects if `datasets/data_1/` exists
- If not, extracts from `datasets/data_1.zip`
- Splits into train/validation (80/20 by default)
- Applies data augmentation to training set

---

## Training

```bash
python scripts/train.py --dataset datasets/data_1 --config configs/model_config.yaml --epochs 50 --batch-size 64
```

**Arguments:**
- `--dataset`: Path to dataset directory (required)
- `--config`: Path to model configuration YAML file (required)
- `--epochs`: Number of epochs (default: 50)
- `--batch-size`: Batch size (default: 64)
- `--val-split`: Validation ratio (default: 0.2)
- `--checkpoint-dir`: Directory to save checkpoints (default: checkpoints)

**Printed Metrics:**
- Dataset loading time (seconds)
- Number of classes
- Train/val sample counts
- Model parameters, MACs, FLOPs
- Per-epoch train/val loss and accuracy

**Outputs:**
- `checkpoints/best.pth` - Best validation accuracy
- `checkpoints/epoch_N.pth` - Periodic checkpoints
- `logs/` - Training logs

---

## Evaluation

```bash
python scripts/evaluate.py --dataset datasets/data_1 --checkpoint checkpoints/best.pth
```

**Arguments:**
- `--dataset`: Path to dataset directory (required)
- `--checkpoint`: Path to checkpoint file (required)
- `--config`: Path to model configuration YAML (default: configs/model_config.yaml)
- `--batch-size`: Batch size (default: 64)

**Printed Metrics:**
- Dataset loading time
- Overall accuracy and loss
- Per-class accuracy breakdown

---

## Model Configuration

Single model config: `configs/model_config.yaml`

```yaml
model:
  name: AdvancedCNN
  layers:
    - Conv2D: {in_channels: 3, out_channels: 64, kernel_size: 3, padding: 1}
    - BatchNorm2D: {num_features: 64}
    - ReLU
    - MaxPool2D: {kernel_size: 2}
    - Dropout: {p: 0.25}
    # ... more layers
    - Flatten
    - Linear: {in_features: 6272, out_features: 512}
    - ReLU
    - Linear: {in_features: 512, out_features: num_classes}

training:
  optimizer: Adam
  learning_rate: 0.001
  weight_decay: 0.0001

augmentation:
  enabled: true
  flip: true
  rotate: true
  brightness: true
  contrast: true
```

The model automatically adapts to 10 or 100 classes based on the dataset.

---

## Implementation Details

### C++ Backend
- All tensor operations in C++17
- Optimized with compiler flags (`-O3`, `-march=native`)
- OpenMP parallelization for CPU
- Optional CUDA GPU acceleration

### Autograd Engine
- Dynamic computational graph
- Gradient tracking per tensor
- Backward pass with chain rule

### No External ML Libraries
- No NumPy for computations (only OpenCV for image I/O)
- No PyTorch/TensorFlow/JAX
- Custom implementations of all layers and operations

---

## Allowed Libraries

**Used:**
- Python 3.12 standard library
- C++ standard library (C++17)
- OpenCV (image loading and preprocessing only)
- pybind11 (Python bindings)
- PyYAML (config files)
- tqdm (progress bars)

**Not Used:**
- NumPy (for numerical operations)
- Any deep learning framework
- Any automatic differentiation library

---

## Troubleshooting

### "Module 'deepnet_backend' not found"
```bash
make clean && make
```

### "pybind11 not found"
```bash
git clone https://github.com/pybind/pybind11.git
```

### IDE Errors ("Too many errors emitted")
This is a clangd/IntelliSense issue with C++ template-heavy code and CUDA files. These are IDE-only errors and **do not affect compilation**. The code compiles and runs correctly. You can:
- Ignore the IDE errors
- Close the C++ files showing errors
- The `make` build will succeed regardless

### Build Fails on Windows
- Install Visual Studio 2019+ with "Desktop development with C++"
- Ensure you're building for x64 architecture
- Or use MinGW-w64 (64-bit version)

### Build Fails on Linux
```bash
sudo apt install build-essential cmake python3-dev
```

---

## Assignment Compliance

- ✅ Custom framework from scratch
- ✅ C++ backend (+20 bonus points)
- ✅ Tensor with autograd
- ✅ All required layers (Conv, Pool, FC, Activation)
- ✅ Dataset loading with measured time
- ✅ Parameters, MACs, FLOPs calculation
- ✅ Training < 3 hours per dataset
- ✅ Only allowed libraries (no NumPy/PyTorch/etc.)

---

## License

Educational project for GNR638 coursework.
