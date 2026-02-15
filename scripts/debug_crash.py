
import sys
import os
import time
import numpy as np

# Add project root and deepnet to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.join(project_root, 'deepnet'))

try:
    import deepnet_backend as backend
except ImportError as e:
    print(f"Error: Could not import deepnet modules: {e}")
    sys.exit(1)

def debug_crash():
    print("Initializing backend...")
    
    batch_size = 128
    channels = 1
    h, w = 28, 28
    
    # Layers (aligned with mnist_config.yaml)
    # Block 1
    conv1 = backend.Conv2D(channels, 32, 3, 1, 1)
    bn1 = backend.BatchNorm2D(32)
    relu1 = backend.ReLU()
    pool1 = backend.MaxPool2D(2, 2)
    
    # Block 2
    conv2 = backend.Conv2D(32, 64, 3, 1, 1)
    bn2 = backend.BatchNorm2D(64)
    relu2 = backend.ReLU()
    pool2 = backend.MaxPool2D(2, 2)
    
    # Classifier
    flatten = backend.Flatten()
    fc1 = backend.Linear(64 * 7 * 7, 128)
    relu3 = backend.ReLU() # This was missing
    # dropout = backend.Dropout(0.3)
    fc2 = backend.Linear(128, 10)
    
    # Loss
    criterion = backend.CrossEntropyLoss()
    
    # Move to CUDA
    layers = [conv1, bn1, conv2, bn2, fc1, fc2]
    # params = []
    print("Moving layers to CUDA...")
    for l in layers:
        for p in l.parameters():
            p.cuda() # Let forward pass auto-move them
        #     pass 
        # params.extend(l.parameters())
    
    # Explicitly collect params as lists
    params = []
    # for l in [conv1, bn1, conv2, bn2, fc1, fc2]:
    #     params.extend(l.parameters())
    
    # Correct order without dropout
    layers_list = [conv1, bn1, relu1, pool1, conv2, bn2, relu2, pool2, flatten, fc1, relu3, fc2]
    # Only layers with params
    param_layers = [conv1, bn1, conv2, bn2, fc1, fc2]
    for l in param_layers:
        params.extend(l.parameters())
            
    # Optimizer
    optimizer = backend.Adam(params, lr=0.001, weight_decay=0.0001)

    print("Starting Training Loop (10 iterations)...", flush=True)
    
    for i in range(10):
        print(f"Iteration {i+1}/10", flush=True)
        
        # Create input (simulating data loader)
        # Re-create access to force new memory allocation potentially
        input_data = np.random.randn(batch_size * channels * h * w).astype(np.float32).tolist()
        x = backend.Tensor.from_data(input_data, [batch_size, channels, h, w], requires_grad=False, cuda=True)
        
        targets_list = [np.random.randint(0, 10) for _ in range(batch_size)]
        
        # Forward
        # Block 1
        print("Block 1 Forward", flush=True)
        x = conv1.forward(x)
        x = bn1.forward(x)
        x = relu1.forward(x)
        x = pool1.forward(x)
        
        # Block 2
        print("Block 2 Forward", flush=True)
        x = conv2.forward(x)
        x = bn2.forward(x)
        x = relu2.forward(x)
        x = pool2.forward(x)
        
        # Classifier
        print("Classifier Forward")
        x = flatten.forward(x)
        x = fc1.forward(x)
        x = relu3.forward(x)
        # x = dropout.forward(x)
        logits = fc2.forward(x)
        
        # Loss
        print("Loss Forward", flush=True)
        loss = criterion.forward(logits, targets_list)
        
        # Force sync by reading data
        print(f"Loss value: {loss.data[0]}", flush=True)
        
        # Backward
        # Backward - Manual Propagation (Same as train.py)
        # 0. Zero Grad (Critical test)
        optimizer.zero_grad()

        # 1. Get grad from loss
        print("Get Input Grad")
        input_grad = criterion.get_input_grad()
        
        # 2. Propagate reverse
        grad = input_grad
        
        # FC2
        print("FC2 Backward")
        grad = fc2.backward(grad)
        
        # Dropout
        # print("Dropout Backward")
        # grad = dropout.backward(grad)
        
        # ReLU 3
        print("ReLU3 Backward")
        grad = relu3.backward(grad)
        
        # FC1
        print("FC1 Backward")
        grad = fc1.backward(grad)
        
        # Flatten
        print("Flatten Backward")
        grad = flatten.backward(grad)
        
        # Block 2
        print("Block 2 Backward")
        grad = pool2.backward(grad)
        grad = relu2.backward(grad)
        grad = bn2.backward(grad)
        grad = conv2.backward(grad)
        
        # Block 1
        print("Block 1 Backward")
        grad = pool1.backward(grad)
        grad = relu1.backward(grad)
        grad = bn1.backward(grad)
        grad = conv1.backward(grad)
        
        # Step
        print("Optimizer Step")
        optimizer.step()
        
        # Sync Logits (like train.py accuracy calc)
        print(f"Logits check loop...", flush=True)
        # Simulate exact access pattern of train.py
        num_classes = 10
        out_data = logits.data # Access once? or inside loop? in train.py: key=lambda j: outputs.data[...]
        # To perfectly match train.py, we should access property every time if that's what python does,
        # but outputs.data usually returns a list/buffer.
        # Let's read all elements.
        for b_idx in range(batch_size):
            for c_idx in range(num_classes):
                _ = logits.data[b_idx * num_classes + c_idx]
        print("Logits check passed.", flush=True)
        
        print("Delete Tensors")
        
        # Explicitly delete tensors to force free
        del x, logits, loss, grad, input_grad
        
    print("SUCCESS: Loop Completed.")

if __name__ == "__main__":
    debug_crash()
