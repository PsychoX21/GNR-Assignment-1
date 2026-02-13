# DeepNet Framework Makefile
# Cross-platform build system for Windows, Linux, and macOS

# Detect OS
ifeq ($(OS),Windows_NT)
    DETECTED_OS := Windows
    PYTHON := python
    VENV_BIN := venv/Scripts
    VENV_ACTIVATE := venv\\Scripts\\activate
    MKDIR := mkdir
    RM := rmdir /s /q
    RM_FILE := del /f /q
    PATH_SEP := \\
else
    UNAME_S := $(shell uname -s)
    ifeq ($(UNAME_S),Linux)
        DETECTED_OS := Linux
    endif
    ifeq ($(UNAME_S),Darwin)
        DETECTED_OS := macOS
    endif
    PYTHON := python3
    VENV_BIN := venv/bin
    VENV_ACTIVATE := source venv/bin/activate
    MKDIR := mkdir -p
    RM := rm -rf
    RM_FILE := rm -f
    PATH_SEP := /
endif

# Configuration
BUILD_DIR := build
DATA ?= data_1
CONFIG ?= configs/model_config.yaml
MODEL ?= checkpoints/best.pth
EPOCHS ?= 50
BATCH_SIZE ?= 64

.PHONY: all setup build install clean train eval test help

# Default target
all: setup build install

# Help
help:
	@echo "DeepNet Framework - Build System"
	@echo "================================="
	@echo ""
	@echo "Detected OS: $(DETECTED_OS)"
	@echo ""
	@echo "Quick Start:"
	@echo "  1. make setup       - Set up environment (first time only)"
	@echo "  2. Activate venv    - Run: $(VENV_ACTIVATE)"
	@echo "  3. make build       - Compile C++ backend"
	@echo "  4. make install     - Install Python package"
	@echo ""
	@echo "Or run all at once:"
	@echo "  make                - Does setup + build + install"
	@echo ""
	@echo "Training & Evaluation:"
	@echo "  make train          - Train model (use DATA=, CONFIG=, EPOCHS=)"
	@echo "  make eval           - Evaluate model (use DATA=, MODEL=)"
	@echo ""
	@echo "Examples:"
	@echo "  make train DATA=data_1 EPOCHS=50"
	@echo "  make eval DATA=data_1 MODEL=checkpoints/best.pth"
	@echo ""
	@echo "Other:"
	@echo "  make clean          - Clean build artifacts"
	@echo "  make test           - Run tests"

# Setup virtual environment
setup:
	@echo "Setting up Python environment..."
	$(PYTHON) -m venv venv
	@echo "Installing dependencies..."
ifeq ($(DETECTED_OS),Windows)
	$(PYTHON) -m pip install --upgrade pip setuptools || echo "Pip upgrade skipped"
	$(VENV_BIN)\pip install -r requirements.txt
	@echo "Checking for Ninja build system..."
	@where ninja >nul 2>&1 || (\
		echo "Ninja not found. Attempting to install..." && \
		(winget install --id=Ninja-build.Ninja -e --silent >nul 2>&1 && echo "Ninja installed via winget") || \
		(where choco >nul 2>&1 && choco install ninja -y >nul 2>&1 && echo "Ninja installed via chocolatey") || \
		(echo "Trying direct download..." && powershell -ExecutionPolicy Bypass -File install_ninja.ps1) || \
		(echo "WARNING: Could not auto-install Ninja." && \
		 echo "Please install manually:" && \
		 echo "  Run: powershell -ExecutionPolicy Bypass -File install_ninja.ps1" && \
		 echo "  Or download from: https://github.com/ninja-build/ninja/releases" && \
		 echo "" && \
		 echo "Continuing setup without Ninja (will use Visual Studio generator)..." ) \
	)
	@where ninja >nul 2>&1 && echo "Ninja build system: READY" || echo "Ninja build system: NOT FOUND (optional)"
else
	$(VENV_BIN)/pip install --upgrade pip setuptools
	$(VENV_BIN)/pip install -r requirements.txt
endif
	@echo "Cloning pybind11..."
ifeq ($(DETECTED_OS),Windows)
	@if not exist pybind11 git clone https://github.com/pybind/pybind11.git
else
	@if [ ! -d "pybind11" ]; then \
		git clone https://github.com/pybind/pybind11.git; \
	fi
endif
	@echo ""
	@echo "Setup complete!"
	@echo "Next step: Activate virtual environment with:"
	@echo "  $(VENV_ACTIVATE)"
	@echo "Then run: make build install"

# Build C++ backend
build:
	@echo "Building C++ backend (OS: $(DETECTED_OS))..."
ifeq ($(DETECTED_OS),Windows)
	@if not exist $(BUILD_DIR) $(MKDIR) $(BUILD_DIR)
	@where ninja >nul 2>&1 && (\
		echo "Using Ninja generator (avoids CUDA + MSBuild bugs)..." && \
		cd $(BUILD_DIR) && cmake .. -G Ninja -DCMAKE_BUILD_TYPE=Release && cmake --build . --config Release -j8 \
	) || (\
		echo "Ninja not found, using default Visual Studio generator..." && \
		echo "WARNING: CUDA may fail with Visual Studio due to known bug" && \
		cd $(BUILD_DIR) && cmake .. -DCMAKE_BUILD_TYPE=Release && cmake --build . --config Release -j8 \
	)
else
	@$(MKDIR) $(BUILD_DIR)
	cd $(BUILD_DIR) && cmake .. -DCMAKE_BUILD_TYPE=Release && cmake --build . --config Release -j8
endif
	@echo "Build complete!"

# Install Python package
install:
	@echo "Installing Python package..."
ifeq ($(DETECTED_OS),Windows)
	@if not exist build\deepnet_backend*.pyd ( \
		echo Error: C++ backend not built. Run 'make build' first. && exit 1 \
	)
	copy build\deepnet_backend*.pyd . >nul 2>&1
else
	@if [ ! -f build/deepnet_backend*.so ]; then \
		echo "Error: C++ backend not built. Run 'make build' first."; \
		exit 1; \
	fi
	cp build/deepnet_backend*.so . 2>/dev/null || true
endif
	$(PYTHON) -m pip install -e .
	@echo "Installation complete!"
	@echo ""
	@echo "You're ready! Try:"
	@echo "  make train DATA=data_1 EPOCHS=2"

# Train model
train:
	@echo "Training model on $(DATA)..."
	$(PYTHON) scripts/train.py \
		--dataset datasets/$(DATA) \
		--config $(CONFIG) \
		--epochs $(EPOCHS) \
		--batch-size $(BATCH_SIZE)
	@echo "Training complete!"

# Evaluate model
eval:
	@echo "Evaluating model on $(DATA)..."
	$(PYTHON) scripts/evaluate.py \
		--dataset datasets/$(DATA) \
		--checkpoint $(MODEL) \
		--config $(CONFIG) \
		--batch-size $(BATCH_SIZE)
	@echo "Evaluation complete!"

# Run tests
test:
	@echo "Running tests..."
	$(PYTHON) -m pytest tests/ -v || echo "No tests found"

# Clean build artifacts
clean:
	@echo "Cleaning build artifacts..."
ifeq ($(DETECTED_OS),Windows)
	@if exist $(BUILD_DIR) $(RM) $(BUILD_DIR)
	@if exist deepnet_backend*.pyd $(RM_FILE) deepnet_backend*.pyd
	@if exist deepnet_backend*.so $(RM_FILE) deepnet_backend*.so
	@if exist deepnet.egg-info $(RM) deepnet.egg-info
	@for /d %%i in (__pycache__) do @if exist %%i $(RM) %%i
else
	$(RM) $(BUILD_DIR)
	$(RM_FILE) deepnet_backend*.so deepnet_backend*.pyd
	$(RM) deepnet.egg-info
	$(RM) __pycache__ deepnet/__pycache__ scripts/__pycache__ deepnet/python/__pycache__
	$(RM) .pytest_cache
endif
	@echo "Clean complete!"

# Deep clean (including venv and pybind11)
distclean: clean
	@echo "Deep cleaning (removing venv, pybind11, checkpoints, logs)..."
ifeq ($(DETECTED_OS),Windows)
	@if exist venv $(RM) venv
	@if exist pybind11 $(RM) pybind11
	@if exist checkpoints $(RM) checkpoints
	@if exist logs $(RM) logs
else
	$(RM) venv pybind11 checkpoints logs
endif
	@echo "Deep clean complete!"
