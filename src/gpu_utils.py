import torch
import platform
import warnings
from typing import Union, Optional, Tuple
from contextlib import contextmanager

class DeviceManager:
    """Smart device management for Mac GPU development"""
    
    def __init__(self):
        self._device = None
        self._device_type = None
        self._initialize_device()
    
    def _initialize_device(self):
        """Initialize the best available device"""
        if platform.system() == 'Darwin' and torch.backends.mps.is_available():
            if torch.backends.mps.is_built():
                self._device = torch.device('mps')
                self._device_type = 'mps'
            else:
                warnings.warn("MPS is available but not built correctly, falling back to CPU")
                self._device = torch.device('cpu')
                self._device_type = 'cpu'
        elif torch.cuda.is_available():
            self._device = torch.device('cuda')
            self._device_type = 'cuda'
        else:
            self._device = torch.device('cpu')
            self._device_type = 'cpu'
    
    @property
    def device(self) -> torch.device:
        """Get the current device"""
        return self._device
    
    @property
    def device_type(self) -> str:
        """Get the device type string"""
        return self._device_type
    
    @property
    def is_gpu(self) -> bool:
        """Check if using GPU"""
        return self._device_type in ['mps', 'cuda']
    
    @property
    def is_apple_silicon(self) -> bool:
        """Check if running on Apple Silicon"""
        return platform.machine() == 'arm64' and platform.system() == 'Darwin'
    
    def to_device(self, tensor_or_model):
        """Move tensor or model to the optimal device"""
        return tensor_or_model.to(self._device)
    
    def synchronize(self):
        """Synchronize device operations"""
        if self._device_type == 'mps':
            torch.mps.synchronize()
        elif self._device_type == 'cuda':
            torch.cuda.synchronize()
    
    def empty_cache(self):
        """Clear device cache"""
        if self._device_type == 'mps':
            torch.mps.empty_cache()
        elif self._device_type == 'cuda':
            torch.cuda.empty_cache()
    
    def get_memory_info(self) -> dict:
        """Get device memory information"""
        if self._device_type == 'mps':
            return {
                'allocated': torch.mps.current_allocated_memory(),
                'cached': torch.mps.current_allocated_memory(),  # MPS doesn't separate these
                'device': 'mps'
            }
        elif self._device_type == 'cuda':
            return {
                'allocated': torch.cuda.memory_allocated(),
                'cached': torch.cuda.memory_reserved(),
                'device': 'cuda'
            }
        else:
            return {
                'allocated': 0,
                'cached': 0,
                'device': 'cpu'
            }


# Global device manager instance
device_manager = DeviceManager()

# Convenience functions
def get_device() -> torch.device:
    """Get the optimal device"""
    return device_manager.device

def to_device(tensor_or_model):
    """Move to optimal device"""
    return device_manager.to_device(tensor_or_model)

def is_gpu_available() -> bool:
    """Check if GPU is available"""
    return device_manager.is_gpu

def synchronize():
    """Synchronize device operations"""
    device_manager.synchronize()

def empty_cache():
    """Clear device cache"""
    device_manager.empty_cache()


@contextmanager
def autocast_context(enabled: bool = True, dtype: torch.dtype = torch.float16):
    """Context manager for automatic mixed precision"""
    device_type = device_manager.device_type
    
    if enabled and device_type in ['mps', 'cuda']:
        with torch.autocast(device_type=device_type, dtype=dtype):
            yield
    else:
        yield


class GPUOptimizedModel:
    """Base class for GPU-optimized models"""
    
    def __init__(self, model, use_mixed_precision: bool = True):
        self.model = model
        self.device_manager = device_manager
        self.use_mixed_precision = use_mixed_precision
        
        # Move model to device
        self.model = self.model.to(self.device_manager.device)
        
        # Setup mixed precision if requested
        if use_mixed_precision and self.device_manager.is_gpu:
            self._setup_mixed_precision()
    
    def _setup_mixed_precision(self):
        """Setup mixed precision training"""
        if self.device_manager.device_type == 'mps':
            # For MPS, we can use autocast
            self.scaler = None  # MPS doesn't need gradient scaling
        elif self.device_manager.device_type == 'cuda':
            self.scaler = torch.cuda.amp.GradScaler()
    
    def forward(self, *args, **kwargs):
        """Forward pass with automatic mixed precision"""
        if self.use_mixed_precision and self.device_manager.is_gpu:
            with autocast_context():
                return self.model(*args, **kwargs)
        else:
            return self.model(*args, **kwargs)
    
    def training_step(self, batch, optimizer, criterion):
        """Optimized training step"""
        # Move batch to device
        if isinstance(batch, (list, tuple)):
            batch = [item.to(self.device_manager.device) if hasattr(item, 'to') else item for item in batch]
        else:
            batch = batch.to(self.device_manager.device)
        
        optimizer.zero_grad()
        
        if self.use_mixed_precision and self.device_manager.is_gpu:
            with autocast_context():
                loss = self._compute_loss(batch, criterion)
            
            if self.scaler:
                self.scaler.scale(loss).backward()
                self.scaler.step(optimizer)
                self.scaler.update()
            else:
                loss.backward()
                optimizer.step()
        else:
            loss = self._compute_loss(batch, criterion)
            loss.backward()
            optimizer.step()
        
        return loss
    
    def _compute_loss(self, batch, criterion):
        """Compute loss - to be implemented by subclasses"""
        raise NotImplementedError


def optimize_for_mac(model, use_mixed_precision: bool = True):
    """Optimize a model for Mac GPU usage"""
    return GPUOptimizedModel(model, use_mixed_precision)


# Memory management utilities
def print_memory_usage():
    """Print current memory usage"""
    memory_info = device_manager.get_memory_info()
    print(f"Device: {memory_info['device']}")
    print(f"Allocated: {memory_info['allocated'] / 1024 / 1024:.1f} MB")
    print(f"Cached: {memory_info['cached'] / 1024 / 1024:.1f} MB")


def cleanup_memory():
    """Clean up GPU memory"""
    import gc
    gc.collect()
    empty_cache()
    print("Memory cleanup completed")


# Performance monitoring
class PerformanceMonitor:
    """Monitor performance of GPU operations"""
    
    def __init__(self):
        self.timings = []
    
    def __enter__(self):
        self.start_time = torch.cuda.Event(enable_timing=True) if device_manager.device_type == 'cuda' else None
        self.end_time = torch.cuda.Event(enable_timing=True) if device_manager.device_type == 'cuda' else None
        
        if self.start_time:
            self.start_time.record()
        else:
            import time
            self.start_time = time.time()
        
        return self
    
    def __exit__(self, *args):
        if device_manager.device_type == 'cuda' and self.end_time:
            self.end_time.record()
            torch.cuda.synchronize()
            elapsed = self.start_time.elapsed_time(self.end_time)
        else:
            device_manager.synchronize()
            import time
            elapsed = (time.time() - self.start_time) * 1000  # Convert to ms
        
        self.timings.append(elapsed)
    
    def average_time(self) -> float:
        """Get average timing in milliseconds"""
        return sum(self.timings) / len(self.timings) if self.timings else 0
    
    def reset(self):
        """Reset timings"""
        self.timings.clear()


# Export commonly used items
__all__ = [
    'DeviceManager', 'device_manager', 'get_device', 'to_device',
    'is_gpu_available', 'synchronize', 'empty_cache', 'autocast_context',
    'GPUOptimizedModel', 'optimize_for_mac', 'print_memory_usage',
    'cleanup_memory', 'PerformanceMonitor'
]
