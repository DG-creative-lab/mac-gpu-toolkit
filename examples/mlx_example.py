"""
MLX example - Apple's native ML framework optimized for Apple Silicon
"""
import time
import numpy as np
from pathlib import Path
import sys

def setup_mlx():
    """Setup and verify MLX installation"""
    print("=== MLX Setup ===")
    
    try:
        import mlx.core as mx
        import mlx.nn as nn
        import mlx.optimizers as optim
        
        print("✅ MLX imported successfully")
        print(f"MLX default device: {mx.default_device()}")
        
        if mx.metal.is_available():
            print("✅ Metal acceleration available")
            return True
        else:
            print("❌ Metal acceleration not available")
            return False
            
    except ImportError as e:
        print(f"❌ MLX import failed: {e}")
        print("💡 Install MLX with: pip install mlx")
        return False

def basic_operations_example():
    """Demonstrate basic MLX operations"""
    print("\n=== Basic MLX Operations ===")
    
    import mlx.core as mx
    
    # Create arrays
    print("🔄 Creating arrays and performing operations...")
    
    # Basic array operations
    x = mx.array([1, 2, 3, 4, 5], dtype=mx.float32)
    y = mx.array([2, 3, 4, 5, 6], dtype=mx.float32)
    
    print(f"x: {x}")
    print(f"y: {y}")
    
    # Arithmetic operations
    z = x + y
    mx.eval(z)  # Ensure computation completes
    print(f"x + y: {z}")
    
    # Matrix operations
    A = mx.random.normal([1000, 1000])
    B = mx.random.normal([1000, 1000])
    
    print("🔄 Matrix multiplication benchmark...")
    start_time = time.time()
    C = mx.matmul(A, B)
    mx.eval(C)  # Force evaluation
    end_time = time.time()
    
    print(f"Matrix multiplication (1000x1000): {end_time - start_time:.4f}s")
    print(f"Result shape: {C.shape}")

def neural_network_example():
    """Create and train a simple neural network with MLX"""
    print("\n=== MLX Neural Network Example ===")
    
    import mlx.core as mx
    import mlx.nn as nn
    import mlx.optimizers as optim
    
    class SimpleNet(nn.Module):
        def __init__(self, input_size, hidden_size, output_size):
            super().__init__()
            self.layers = [
                nn.Linear(input_size, hidden_size),
                nn.Linear(hidden_size, hidden_size),
                nn.Linear(hidden_size, output_size)
            ]
        
        def __call__(self, x):
            for i, layer in enumerate(self.layers):
                x = layer(x)
                if i < len(self.layers) - 1:  # Don't apply activation to last layer
                    x = nn.relu(x)
            return x
    
    # Create model
    model = SimpleNet(784, 256, 10)  # MNIST-like dimensions
    
    # Generate synthetic data
    print("📊 Generating synthetic training data...")
    batch_size = 64
    X = mx.random.normal([batch_size, 784])
    y = mx.random.randint(0, 10, [batch_size])
    
    # Convert labels to one-hot
    y_onehot = mx.eye(10)[y]
    
    # Define loss function
    def cross_entropy_loss(logits, targets):
        log_probs = nn.log_softmax(logits, axis=-1)
        return -mx.mean(mx.sum(targets * log_probs, axis=-1))
    
    # Create optimizer
    optimizer = optim.Adam(learning_rate=0.001)
    
    # Training function
    def loss_fn(model, X, y):
        logits = model(X)
        return cross_entropy_loss(logits, y)
    
    # Get loss and gradients function
    loss_and_grad_fn = nn.value_and_grad(model, loss_fn)
    
    # Training loop
    print("🔄 Training neural network...")
    
    num_epochs = 5
    for epoch in range(num_epochs):
        start_time = time.time()
        
        # Forward pass and compute gradients
        loss, grads = loss_and_grad_fn(model, X, y_onehot)
        
        # Update parameters
        optimizer.update(model, grads)
        
        # Force evaluation
        mx.eval(model.parameters(), optimizer.state)
        
        epoch_time = time.time() - start_time
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss.item():.4f}, Time: {epoch_time:.4f}s")
    
    # Test inference
    print("🔄 Testing inference...")
    test_input = mx.random.normal([10, 784])
    
    start_time = time.time()
    predictions = model(test_input)
    mx.eval(predictions)
    inference_time = time.time() - start_time
    
    print(f"Inference on 10 samples: {inference_time:.4f}s")
    print(f"Predictions shape: {predictions.shape}")

def performance_comparison():
    """Compare MLX performance with NumPy"""
    print("\n=== MLX vs NumPy Performance Comparison ===")
    
    import mlx.core as mx
    
    sizes = [500, 1000, 2000]
    
    for size in sizes:
        print(f"\nMatrix size: {size}x{size}")
        
        # NumPy benchmark
        np_a = np.random.randn(size, size).astype(np.float32)
        np_b = np.random.randn(size, size).astype(np.float32)
        
        start_time = time.time()
        np_result = np.matmul(np_a, np_b)
        numpy_time = time.time() - start_time
        
        print(f"NumPy: {numpy_time:.4f}s")
        
        # MLX benchmark
        mx_a = mx.array(np_a)
        mx_b = mx.array(np_b)
        
        start_time = time.time()
        mx_result = mx.matmul(mx_a, mx_b)
        mx.eval(mx_result)  # Ensure computation completes
        mlx_time = time.time() - start_time
        
        print(f"MLX: {mlx_time:.4f}s")
        
        if numpy_time > mlx_time:
            speedup = numpy_time / mlx_time
            print(f"🚀 MLX speedup: {speedup:.2f}x")
        else:
            slowdown = mlx_time / numpy_time
            print(f"⚠️  MLX slowdown: {slowdown:.2f}x")

def memory_efficiency_example():
    """Demonstrate MLX memory efficiency"""
    print("\n=== MLX Memory Efficiency ===")
    
    import mlx.core as mx
    
    print("🔄 Testing lazy evaluation...")
    
    # Create large arrays without immediately computing
    x = mx.random.normal([2000, 2000])
    y = mx.random.normal([2000, 2000])
    
    # Chain operations (these are lazy)
    z = x + y
    w = mx.matmul(z, x)
    result = mx.mean(w)
    
    print("Operations defined (lazy evaluation)")
    
    # Force evaluation
    start_time = time.time()
    mx.eval(result)
    eval_time = time.time() - start_time
    
    print(f"Evaluation completed in {eval_time:.4f}s")
    print(f"Final result: {result.item():.4f}")

def main():
    """Run all MLX examples"""
    print("MLX Examples for Apple Silicon")
    print("=" * 50)
    
    # Setup MLX
    if not setup_mlx():
        print("❌ MLX setup failed. Cannot run examples.")
        return
    
    # Run examples
    basic_operations_example()
    neural_network_example()
    performance_comparison()
    memory_efficiency_example()
    
    print("\n✅ All MLX examples completed!")

if __name__ == "__main__":
    main()