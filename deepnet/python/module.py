"""PyTorch-like Module abstraction"""

import sys
import os

# Add build directory to path
build_path = os.path.join(os.path.dirname(__file__), '../../build')
if os.path.exists(build_path):
    sys.path.insert(0, build_path)

try:
    import deepnet_backend as backend
except ImportError:
    print("ERROR: Could not import deepnet_backend. Please run 'make build install' first.")
    sys.exit(1)

class Module:
    """Base class for all neural network modules"""
    
    def __init__(self):
        self._modules = {}
        self._parameters = []
        self._training = True
    
    def forward(self, *args, **kwargs):
        """Forward pass - must be implemented by subclasses"""
        raise NotImplementedError("Subclasses must implement forward()")
    
    def __call__(self, *args, **kwargs):
        """Make module callable"""
        return self.forward(*args, **kwargs)
    
    def add_module(self, name, module):
        """Add a child module"""
        self._modules[name] = module
        setattr(self, name, module)
    
    def parameters(self):
        """Get all parameters recursively"""
        params = []
        # Get own parameters
        params.extend(self._parameters)
        # Get parameters from child modules
        for module in self._modules.values():
            if isinstance(module, Module):
                params.extend(module.parameters())
            elif hasattr(module, 'parameters'):
                params.extend(module.parameters())
        return params
    
    def train(self):
        """Set module to training mode"""
        self._training = True
        for module in self._modules.values():
            if isinstance(module, Module):
                module.train()
            elif hasattr(module, 'train'):
                module.train()
    
    def eval(self):
        """Set module to evaluation mode"""
        self._training = False
        for module in self._modules.values():
            if isinstance(module, Module):
                module.eval()
            elif hasattr(module, 'eval'):
                module.eval()
    
    def zero_grad(self):
        """Zero all parameter gradients"""
        for param in self.parameters():
            param.zero_grad()
    
    def state_dict(self):
        """Get state dictionary for saving"""
        state = {}
        for i, param in enumerate(self.parameters()):
            state[f'param_{i}'] = {
                'data': param.data,
                'shape': param.shape,
                'requires_grad': param.requires_grad
            }
        return state
    
    def load_state_dict(self, state):
        """Load state dictionary"""
        params = self.parameters()
        for i, param in enumerate(params):
            if f'param_{i}' in state:
                param.data = state[f'param_{i}']['data']


class Sequential(Module):
    """Sequential container for layers"""
    
    def __init__(self, *layers):
        super().__init__()
        self.layers = list(layers)
        for i, layer in enumerate(self.layers):
            self.add_module(f'layer_{i}', layer)
    
    def forward(self, x):
        """Forward pass through all layers"""
        for layer in self.layers:
            if isinstance(layer, Module):
                x = layer(x)
            elif hasattr(layer, 'forward'):
                x = layer.forward(x)
            else:
                raise ValueError(f"Invalid layer type: {type(layer)}")
        return x
    
    def backward(self, grad):
        """Backward pass through all layers in reverse"""
        for layer in reversed(self.layers):
            if isinstance(layer, Module):
                grad = layer.backward(grad)
            elif hasattr(layer, 'backward'):
                grad = layer.backward(grad)
        return grad
    
    def parameters(self):
        """Get all parameters from all layers"""
        params = []
        for layer in self.layers:
            if isinstance(layer, Module):
                params.extend(layer.parameters())
            elif hasattr(layer, 'parameters'):
                layer_params = layer.parameters()
                if layer_params:
                    params.extend(layer_params)
        return params


# Wrapper classes for backend layers
class Conv2DWrapper(Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=True):
        super().__init__()
        self.layer = backend.Conv2D(in_channels, out_channels, kernel_size, stride, padding, bias)
        self._parameters = self.layer.parameters()
    
    def forward(self, x):
        return self.layer.forward(x)
    
    def backward(self, grad):
        return self.layer.backward(grad)


class LinearWrapper(Module):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.layer = backend.Linear(in_features, out_features, bias)
        self._parameters = self.layer.parameters()
    
    def forward(self, x):
        return self.layer.forward(x)
    
    def backward(self, grad):
        return self.layer.backward(grad)


class ReLUWrapper(Module):
    def __init__(self):
        super().__init__()
        self.layer = backend.ReLU()
    
    def forward(self, x):
        return self.layer.forward(x)
    
    def backward(self, grad):
        return self.layer.backward(grad)


class MaxPool2DWrapper(Module):
    def __init__(self, kernel_size, stride=None):
        super().__init__()
        if stride is None:
            stride = kernel_size
        self.layer = backend.MaxPool2D(kernel_size, stride)
    
    def forward(self, x):
        return self.layer.forward(x)
    
    def backward(self, grad):
        return self.layer.backward(grad)


class BatchNorm2DWrapper(Module):
    def __init__(self, num_features, eps=1e-5, momentum=0.1):
        super().__init__()
        self.layer = backend.BatchNorm2D(num_features, eps, momentum)
        self._parameters = self.layer.parameters()
    
    def forward(self, x):
        return self.layer.forward(x)
    
    def backward(self, grad):
        return self.layer.backward(grad)
    
    def train(self):
        self._training = True
        self.layer.train()
    
    def eval(self):
        self._training = False
        self.layer.eval()


class BatchNorm1DWrapper(Module):
    def __init__(self, num_features, eps=1e-5, momentum=0.1):
        super().__init__()
        self.layer = backend.BatchNorm1D(num_features, eps, momentum)
        self._parameters = self.layer.parameters()
    
    def forward(self, x):
        return self.layer.forward(x)
    
    def backward(self, grad):
        return self.layer.backward(grad)
    
    def train(self):
        self._training = True
        self.layer.train()
    
    def eval(self):
        self._training = False
        self.layer.eval()


class DropoutWrapper(Module):
    def __init__(self, p=0.5):
        super().__init__()
        self.layer = backend.Dropout(p)
    
    def forward(self, x):
        return self.layer.forward(x)
    
    def backward(self, grad):
        return self.layer.backward(grad)
    
    def train(self):
        self._training = True
        self.layer.train()
    
    def eval(self):
        self._training = False
        self.layer.eval()


class FlattenWrapper(Module):
    def __init__(self, start_dim=1, end_dim=-1):
        super().__init__()
        self.layer = backend.Flatten(start_dim, end_dim)
    
    def forward(self, x):
        return self.layer.forward(x)
    
    def backward(self, grad):
        return self.layer.backward(grad)


class LeakyReLUWrapper(Module):
    def __init__(self, negative_slope=0.01):
        super().__init__()
        self.layer = backend.LeakyReLU(negative_slope)
    
    def forward(self, x):
        return self.layer.forward(x)
    
    def backward(self, grad):
        return self.layer.backward(grad)


class TanhWrapper(Module):
    def __init__(self):
        super().__init__()
        self.layer = backend.Tanh()
    
    def forward(self, x):
        return self.layer.forward(x)
    
    def backward(self, grad):
        return self.layer.backward(grad)


class SigmoidWrapper(Module):
    def __init__(self):
        super().__init__()
        self.layer = backend.Sigmoid()
    
    def forward(self, x):
        return self.layer.forward(x)
    
    def backward(self, grad):
        return self.layer.backward(grad)


class AvgPool2DWrapper(Module):
    def __init__(self, kernel_size, stride=None):
        super().__init__()
        if stride is None:
            stride = kernel_size
        self.layer = backend.AvgPool2D(kernel_size, stride)
    
    def forward(self, x):
        return self.layer.forward(x)
    
    def backward(self, grad):
        return self.layer.backward(grad)
