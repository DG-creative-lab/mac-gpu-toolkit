import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import time
from pathlib import Path
import sys

# Add src to path so we can import our utilities
sys.path.append(str(Path(__file__).parent.parent / 'src'))
from gpu_utils import device_manager, autocast_context, PerformanceMonitor, cleanup_memory

class ConvNet(nn.Module):
    """Simple CNN for demonstration"""
    
    def __init__(self, num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.5)
        
        self.fc1 = nn.Linear(128 * 4 * 4, 256)
        self.fc2 = nn.Linear(256, num_classes)
    
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        
        x = x.view(x.size(0), -1)
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.fc2(x)
        return x

def generate_synthetic_data(batch_size=64, num_batches=50):
    """Generate synthetic CIFAR-10 like data"""
    print(f"📊 Generating {num_batches} batches of size {batch_size}")
    
    data = []
    for _ in range(num_batches):
        # Create random images (3, 32, 32) and labels
        images = torch.randn(batch_size, 3, 32, 32)
        labels = torch.randint(0, 10, (batch_size,))
        data.append((images, labels))
    
    return data

def train_model_basic():
    """Basic training example without optimization"""
    print("=== Basic PyTorch Training (No Optimization) ===")
    
    # Model setup
    model = ConvNet().to(device_manager.device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Data
    train_data = generate_synthetic_data(batch_size=32, num_batches=20)
    
    # Training loop
    model.train()
    total_time = 0
    
    for epoch in range(2):
        epoch_loss = 0
        epoch_start = time.time()
        
        for batch_idx, (data, target) in enumerate(train_data):
            # Move to device
            data, target = data.to(device_manager.device), target.to(device_manager.device)
            
            # Standard training step
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            
            if batch_idx % 10 == 0:
                print(f"Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item():.4f}")
        
        epoch_time = time.time() - epoch_start
        total_time += epoch_time
        print(f"Epoch {epoch} completed in {epoch_time:.2f}s, Average Loss: {epoch_loss/len(train_data):.4f}")
    
    print(f"Basic training completed in {total_time:.2f}s")
    return total_time

def train_model_optimized():
    """Optimized training with mixed precision and performance monitoring"""
    print("\n=== Optimized PyTorch Training (With GPU Optimization) ===")
    
    # Model setup
    model = ConvNet().to(device_manager.device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Data
    train_data = generate_synthetic_data(batch_size=64, num_batches=20)  # Larger batch size
    
    # Performance monitoring
    monitor = PerformanceMonitor()
    
    # Training loop with optimizations
    model.train()
    total_time = 0
    
    for epoch in range(2):
        epoch_loss = 0
        epoch_start = time.time()
        
        for batch_idx, (data, target) in enumerate(train_data):
            with monitor:
                # Move to device
                data, target = data.to(device_manager.device), target.to(device_manager.device)
                
                # Optimized training step with mixed precision
                optimizer.zero_grad()
                
                with autocast_context(enabled=device_manager.is_gpu, dtype=torch.float16):
                    output = model(data)
                    loss = criterion(output, target)
                
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
            
            if batch_idx % 10 == 0:
                avg_time = monitor.average_time()
                print(f"Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item():.4f}, "
                      f"Avg Batch Time: {avg_time:.2f}ms")
        
        epoch_time = time.time() - epoch_start
        total_time += epoch_time
        print(f"Epoch {epoch} completed in {epoch_time:.2f}s, Average Loss: {epoch_loss/len(train_data):.4f}")
    
    print(f"Optimized training completed in {total_time:.2f}s")
    print(f"Average batch processing time: {monitor.average_time():.2f}ms")
    
    return total_time

def inference_example():
    """Example of optimized inference"""
    print("\n=== Inference Example ===")
    
    # Load trained model (for demo, we'll use a fresh one)
    model = ConvNet().to(device_manager.device)
    model.eval()
    
    # Generate test data
    test_data = torch.randn(100, 3, 32, 32).to(device_manager.device)
    
    print(f"Running inference on {test_data.shape[0]} samples...")
    
    with torch.no_grad():
        start_time = time.time()
        
        # Batch inference with mixed precision
        with autocast_context(enabled=device_manager.is_gpu, dtype=torch.float16):
            predictions = model(test_data)
            predicted_classes = torch.argmax(predictions, dim=1)
        
        device_manager.synchronize()  # Ensure completion
        
        inference_time = time.time() - start_time
    
    print(f"Inference completed in {inference_time:.4f}s")
    print(f"Time per sample: {(inference_time * 1000) / test_data.shape[0]:.2f}ms")
    print(f"Predictions shape: {predictions.shape}")
    print(f"Sample predictions: {predicted_classes[:10].tolist()}")

def memory_usage_example():
    """Demonstrate memory management best practices"""
    print("\n=== Memory Management Example ===")
    
    from gpu_utils import print_memory_usage
    
    print("Initial memory usage:")
    print_memory_usage()
    
    # Allocate some large tensors
    print("\nAllocating large tensors...")
    tensors = []
    for i in range(5):
        tensor = torch.randn(1000, 1000).to(device_manager.device)
        tensors.append(tensor)
        if i % 2 == 0:
            print(f"After allocation {i+1}:")
            print_memory_usage()
    
    print("\nCleaning up memory...")
    tensors.clear()
    cleanup_memory()
    
    print("After cleanup:")
    print_memory_usage()

def main():
    """Run all PyTorch examples"""
    print("PyTorch MPS Examples for Mac GPU")
    print("=" * 50)
    
    # Check if MPS is available
    if not device_manager.is_gpu:
        print("⚠️  GPU not available, examples will run on CPU")
    else:
        print(f"✅ Using device: {device_manager.device}")
    
    # Run examples
    basic_time = train_model_basic()
    optimized_time = train_model_optimized()
    
    # Compare performance
    if basic_time > 0 and optimized_time > 0:
        speedup = basic_time / optimized_time
        print(f"\n🚀 Optimization speedup: {speedup:.2f}x")
    
    # Other examples
    inference_example()
    memory_usage_example()
    
    print("\n✅ All PyTorch examples completed!")

if __name__ == "__main__":
    main()