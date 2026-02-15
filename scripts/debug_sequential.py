
import sys
import os
import numpy as np

# Add project root and deepnet to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.join(project_root, 'deepnet'))

try:
    import deepnet_backend as backend
    from deepnet.python.models import build_model_from_config
    # Test if importing data (and thus cv2) breaks things
    from deepnet.python.data import DataLoader
except ImportError as e:
    print(f"Error: Could not import deepnet modules: {e}")
    sys.exit(1)

def debug_sequential():
    print("Initializing backend...")
    
    batch_size = 128
    channels = 1
    h, w = 28, 28
    num_classes = 10
    
    # Build model using same function as train.py
    config_path = os.path.join(project_root, "configs", "mnist_config.yaml")
    print(f"Building model from {config_path}")
    model, config = build_model_from_config(config_path, num_classes)
    
    # Loss
    criterion = backend.CrossEntropyLoss()
    
    # Optimizer - Create BEFORE cuda() to match train.py
    params = list(model.parameters())
    optimizer = backend.Adam(params, lr=0.001, weight_decay=0.0001)

    # Move to CUDA
    print("Moving model to CUDA (after optimizer creation)...", flush=True)
    model.cuda()
    
    # Verify params match
    # params = list(model.parameters()) # Refetch? Or check original list?
    # In C++, shared_ptr should handle this IF cuda() is in-place.
    # If cuda() replaces the object, then optimizer has stale objects.
    
    print(f"Total params: {len(params)}")
    for i, p in enumerate(params):
        print(f"Param {i}: CUDA={p.is_cuda}, Shape={p.shape}")

    # Scheduler (StepLR) - mimicking train.py
    scheduler = backend.StepLR(0.001, step_size=10, gamma=0.5)
    print("Scheduler initialized: StepLR")

    print("Starting Training Loop (10 iterations)...", flush=True)
    
    for i in range(10):
        print(f"Iteration {i+1}/10", flush=True)
        
        # Create input 
        input_data = np.random.randn(batch_size * channels * h * w).astype(np.float32).tolist()
        x = backend.Tensor.from_data(input_data, [batch_size, channels, h, w], requires_grad=False, cuda=True)
        
        targets_list = [np.random.randint(0, 10) for _ in range(batch_size)]
        
        # Forward
        print("T: Pre-Forward", flush=True)
        outputs = model(x)
        
        # Loss
        print("T: Pre-Loss", flush=True)
        loss_tensor = criterion.forward(outputs, targets_list)
        print("T: Loss Data Access", flush=True)
        loss = loss_tensor.data[0]
        print(f"Loss: {loss}", flush=True)
        
        # Backward
        print("T: Pre-Grad", flush=True)
        input_grad = criterion.get_input_grad()
        print("T: Pre-Backward", flush=True)
        model.backward(input_grad)
        
        # Step
        print("T: Pre-Step", flush=True)
        optimizer.step()
        
        # Scheduler Step
        if scheduler:
            new_lr = scheduler.step() # In train.py this is after validation, but let's test if calling it breaks anything
            # optimizer.set_lr(new_lr) # train.py does this
        
        # Accuracy Check (Simulate train.py)
        print("T: Pre-Accuracy", flush=True)
        for b in range(batch_size):
            # Access outputs.data repeatedly
             _ = max(range(num_classes), key=lambda j: outputs.data[b * num_classes + j])
        
        print("Iteration complete.", flush=True)
        
    print("SUCCESS: Loop Completed.")

if __name__ == "__main__":
    debug_sequential()
