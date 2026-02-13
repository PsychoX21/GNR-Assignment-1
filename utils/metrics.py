"""Utilities for calculating model metrics"""

def count_parameters(model):
    """Count total trainable parameters"""
    total = 0
    params = model.parameters()
    for param in params:
        param_count = 1
        for dim in param.shape:
            param_count *= dim
        total += param_count
    return total


def estimate_macs_flops(model, input_shape):
    """
    Estimate MACs and FLOPs for the model
    
    Args:
        model: The neural network model
        input_shape: Input tensor shape [batch, channels, height, width]
    
    Returns:
        dict: Dictionary with 'macs' and 'flops' estimates
    """
    macs = 0
    flops = 0
    
    # Get model layers
    if hasattr(model, 'layers'):
        layers = model.layers
    else:
        return {'macs': 0, 'flops': 0}
    
    current_shape = list(input_shape)
    
    for layer in layers:
        layer_type = type(layer).__name__
        
        if 'Conv2D' in layer_type:
            # Conv2D: (batch * out_h * out_w * out_c) * (kernel_h * kernel_w * in_c)
            # Simplified calculation
            if hasattr(layer, 'layer'):
                # Access backend layer
                backend_layer = layer.layer
                params = backend_layer.parameters()
                if params:
                    weight = params[0]
                    # Weight shape: [out_c, in_c, kernel_h, kernel_w]
                    out_c = weight.shape[0]
                    in_c = weight.shape[1]
                    kernel_h = weight.shape[2]
                    kernel_w = weight.shape[3]
                    
                    # Assume stride=1, padding=0 for simplicity
                    out_h = current_shape[2] - kernel_h + 1
                    out_w = current_shape[3] - kernel_w + 1
                    
                    layer_macs = (current_shape[0] * out_h * out_w * out_c) * (kernel_h * kernel_w * in_c)
                    macs += layer_macs
                    
                    # Update current shape
                    current_shape[1] = out_c
                    current_shape[2] = out_h
                    current_shape[3] = out_w
        
        elif 'Linear' in layer_type:
            # Linear: batch * in_features * out_features
            if hasattr(layer, 'layer'):
                backend_layer = layer.layer
                params = backend_layer.parameters()
                if params:
                    weight = params[0]
                    # Weight shape: [out_features, in_features]
                    out_features = weight.shape[0]
                    in_features = weight.shape[1]
                    
                    # Flatten all dimensions except batch
                    batch = current_shape[0]
                    layer_macs = batch * in_features * out_features
                    macs += layer_macs
                    
                    # Update shape
                    current_shape = [batch, out_features]
        
        elif 'MaxPool' in layer_type or 'AvgPool' in layer_type:
            # Pooling reduces spatial dimensions
            # Assuming kernel_size = 2, stride = 2
            if len(current_shape) == 4:
                current_shape[2] = current_shape[2] // 2
                current_shape[3] = current_shape[3] // 2
        
        elif 'Flatten' in layer_type:
            # Flatten to 2D
            if len(current_shape) == 4:
                flat_size = current_shape[1] * current_shape[2] * current_shape[3]
                current_shape = [current_shape[0], flat_size]
    
    # FLOPs are approximately 2 * MACs (multiply + add)
    flops = 2 * macs
    
    return {'macs': macs, 'flops': flops}


def print_model_summary(model, input_shape):
    """Print a summary of the model"""
    print("\nModel Summary")
    print("=" * 60)
    
    # Count parameters
    total_params = count_parameters(model)
    print(f"Total Parameters: {total_params:,}")
    
    # Estimate MACs and FLOPs
    metrics = estimate_macs_flops(model, input_shape)
    print(f"Estimated MACs: {metrics['macs']:,}")
    print(f"Estimated FLOPs: {metrics['flops']:,}")
    
    # Model size in MB (assuming float32)
    model_size_mb = (total_params * 4) / (1024 ** 2)
    print(f"Model Size: {model_size_mb:.2f} MB")
    
    print("=" * 60)
