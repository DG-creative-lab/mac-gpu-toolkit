import time
import sys
import numpy as np
from typing import Dict, List, Tuple

class Benchmark:
    def __init__(self):
        self.results = {}
    
    def time_operation(self, func, *args, **kwargs):
        """Time a function execution"""
        # Warm up
        for _ in range(3):
            func(*args, **kwargs)
        
        # Benchmark
        times = []
        for _ in range(10):
            start = time.time()
            result = func(*args, **kwargs)
            
            # Ensure completion for GPU operations
            if hasattr(result, 'device') and 'mps' in str(result.device):
                import torch
                torch.mps.synchronize()
            
            end = time.time()
            times.append(end - start)
        
        return np.mean(times), np.std(times), result

def benchmark_pytorch():
    """Benchmark PyTorch operations"""
    print("=== PyTorch Benchmarks ===")
    
    try:
        import torch
        
        benchmark = Benchmark()
        sizes = [(512, 512), (1024, 1024), (2048, 2048), (4096, 4096)]
        
        for size in sizes:
            print(f"\nMatrix size: {size[0]}x{size[1]}")
            
            # CPU benchmark
            try:
                def cpu_matmul():
                    x = torch.randn(size, device='cpu')
                    y = torch.randn(size, device='cpu')
                    return torch.matmul(x, y)
                
                cpu_time, cpu_std, _ = benchmark.time_operation(cpu_matmul)
                print(f"CPU: {cpu_time:.4f}s ± {cpu_std:.4f}s")
            except Exception as e:
                print(f"CPU failed: {e}")
                continue
            
            # GPU benchmark
            if torch.backends.mps.is_available():
                try:
                    def gpu_matmul():
                        x = torch.randn(size, device='mps')
                        y = torch.randn(size, device='mps')
                        return torch.matmul(x, y)
                    
                    gpu_time, gpu_std, _ = benchmark.time_operation(gpu_matmul)
                    speedup = cpu_time / gpu_time
                    print(f"GPU: {gpu_time:.4f}s ± {gpu_std:.4f}s (speedup: {speedup:.2f}x)")
                except Exception as e:
                    print(f"GPU failed: {e}")
            else:
                print("GPU: Not available")
    
    except ImportError:
        print("PyTorch not available")

def benchmark_neural_networks():
    """Benchmark neural network training"""
    print("\n=== Neural Network Training Benchmarks ===")
    
    try:
        import torch
        import torch.nn as nn
        import torch.optim as optim
        
        class SimpleNet(nn.Module):
            def __init__(self, input_size, hidden_size, output_size):
                super().__init__()
                self.layers = nn.Sequential(
                    nn.Linear(input_size, hidden_size),
                    nn.ReLU(),
                    nn.Linear(hidden_size, hidden_size),
                    nn.ReLU(),
                    nn.Linear(hidden_size, output_size)
                )
            
            def forward(self, x):
                return self.layers(x)
        
        # Test configuration
        batch_size = 256
        input_size = 1024
        hidden_size = 2048
        output_size = 10
        epochs = 5
        
        print(f"Network: {input_size} -> {hidden_size} -> {hidden_size} -> {output_size}")
        print(f"Batch size: {batch_size}, Epochs: {epochs}")
        
        def train_model(device):
            model = SimpleNet(input_size, hidden_size, output_size).to(device)
            optimizer = optim.Adam(model.parameters())
            criterion = nn.CrossEntropyLoss()
            
            # Generate synthetic data
            X = torch.randn(batch_size * 10, input_size, device=device)
            y = torch.randint(0, output_size, (batch_size * 10,), device=device)
            
            start_time = time.time()
            
            for epoch in range(epochs):
                for i in range(0, len(X), batch_size):
                    batch_X = X[i:i+batch_size]
                    batch_y = y[i:i+batch_size]
                    
                    optimizer.zero_grad()
                    outputs = model(batch_X)
                    loss = criterion(outputs, batch_y)
                    loss.backward()
                    optimizer.step()
            
            if device.type == 'mps':
                torch.mps.synchronize()
            
            return time.time() - start_time
        
        # CPU training
        try:
            cpu_time = train_model(torch.device('cpu'))
            print(f"CPU training: {cpu_time:.2f}s")
        except Exception as e:
            print(f"CPU training failed: {e}")
            return
        
        # GPU training
        if torch.backends.mps.is_available():
            try:
                gpu_time = train_model(torch.device('mps'))
                speedup = cpu_time / gpu_time
                print(f"GPU training: {gpu_time:.2f}s (speedup: {speedup:.2f}x)")
            except Exception as e:
                print(f"GPU training failed: {e}")
        else:
            print("GPU training: Not available")
    
    except ImportError:
        print("PyTorch not available")

def benchmark_tensorflow():
    """Benchmark TensorFlow operations"""
    print("\n=== TensorFlow Benchmarks ===")
    
    try:
        import tensorflow as tf
        
        sizes = [(512, 512), (1024, 1024), (2048, 2048)]
        
        for size in sizes:
            print(f"\nMatrix size: {size[0]}x{size[1]}")
            
            # CPU benchmark
            try:
                with tf.device('/CPU:0'):
                    x = tf.random.normal(size)
                    y = tf.random.normal(size)
                    
                    start_time = time.time()
                    for _ in range(10):
                        z = tf.matmul(x, y)
                    cpu_time = (time.time() - start_time) / 10
                    
                print(f"CPU: {cpu_time:.4f}s")
            except Exception as e:
                print(f"CPU failed: {e}")
                continue
            
            # GPU benchmark
            physical_devices = tf.config.list_physical_devices('GPU')
            if physical_devices:
                try:
                    with tf.device('/GPU:0'):
                        x = tf.random.normal(size)
                        y = tf.random.normal(size)
                        
                        start_time = time.time()
                        for _ in range(10):
                            z = tf.matmul(x, y)
                        gpu_time = (time.time() - start_time) / 10
                        
                    speedup = cpu_time / gpu_time
                    print(f"GPU: {gpu_time:.4f}s (speedup: {speedup:.2f}x)")
                except Exception as e:
                    print(f"GPU failed: {e}")
            else:
                print("GPU: Not available")
    
    except ImportError:
        print("TensorFlow not available")

def benchmark_mlx():
    """Benchmark MLX operations"""
    print("\n=== MLX Benchmarks ===")
    
    try:
        import mlx.core as mx
        
        sizes = [(512, 512), (1024, 1024), (2048, 2048)]
        
        for size in sizes:
            print(f"\nMatrix size: {size[0]}x{size[1]}")
            
            try:
                x = mx.random.normal(size)
                y = mx.random.normal(size)
                
                start_time = time.time()
                for _ in range(10):
                    z = mx.matmul(x, y)
                    mx.eval(z)  # Ensure computation completes
                mlx_time = (time.time() - start_time) / 10
                
                print(f"MLX: {mlx_time:.4f}s")
            except Exception as e:
                print(f"MLX failed: {e}")
    
    except ImportError:
        print("MLX not available")

def memory_benchmark():
    """Benchmark memory usage and management"""
    print("\n=== Memory Management Benchmark ===")
    
    try:
        import torch
        
        if not torch.backends.mps.is_available():
            print("MPS not available for memory benchmark")
            return
        
        device = torch.device('mps')
        
        print("Testing memory allocation and cleanup...")
        
        # Initial memory
        initial_memory = torch.mps.current_allocated_memory()
        print(f"Initial memory: {initial_memory / 1024 / 1024:.1f} MB")
        
        # Allocate large tensors
        tensors = []
        for i in range(10):
            tensor = torch.randn(1000, 1000, device=device)
            tensors.append(tensor)
            current_memory = torch.mps.current_allocated_memory()
            print(f"After allocation {i+1}: {current_memory / 1024 / 1024:.1f} MB")
        
        # Clear references
        tensors.clear()
        
        # Force garbage collection
        import gc
        gc.collect()
        torch.mps.empty_cache()
        
        final_memory = torch.mps.current_allocated_memory()
        print(f"After cleanup: {final_memory / 1024 / 1024:.1f} MB")
        
        if abs(final_memory - initial_memory) < 1024 * 1024:  # Within 1MB
            print("✅ Memory management working correctly")
        else:
            print("⚠️  Memory may not be fully released")
    
    except Exception as e:
        print(f"Memory benchmark failed: {e}")

def main():
    """Run all benchmarks"""
    print("Mac GPU Performance Benchmarks")
    print("=" * 50)
    print("This may take several minutes...")
    
    benchmark_pytorch()
    benchmark_neural_networks()
    benchmark_tensorflow()
    benchmark_mlx()
    memory_benchmark()
    
    print("\n" + "=" * 50)
    print("Benchmark completed!")
    print("💡 Results show relative performance between CPU and GPU")
    print("💡 Actual speedups depend on workload characteristics")

if __name__ == "__main__":
    main()