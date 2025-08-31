import tensorflow as tf
import numpy as np
import time
from pathlib import Path
import sys

def setup_tensorflow():
    """Setup TensorFlow for optimal Mac performance"""
    print("=== TensorFlow Setup ===")
    
    # Check TensorFlow version
    print(f"TensorFlow version: {tf.__version__}")
    
    # List available devices
    physical_devices = tf.config.list_physical_devices()
    print("Available devices:")
    for device in physical_devices:
        print(f"  {device}")
    
    # Check GPU availability
    gpu_devices = tf.config.list_physical_devices('GPU')
    if gpu_devices:
        print(f"✅ Found {len(gpu_devices)} GPU device(s)")
        for gpu in gpu_devices:
            print(f"  GPU: {gpu}")
        return True
    else:
        print("❌ No GPU devices found")
        return False

def create_model():
    """Create a simple CNN model"""
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.Dense(10, activation='softmax')
    ])
    
    return model

def generate_synthetic_data(num_samples=1000, batch_size=32):
    """Generate synthetic CIFAR-10 like data"""
    print(f"📊 Generating {num_samples} synthetic samples")
    
    # Generate random images and labels
    X = np.random.randn(num_samples, 32, 32, 3).astype(np.float32)
    y = np.random.randint(0, 10, num_samples)
    y = tf.keras.utils.to_categorical(y, 10)
    
    return X, y

def train_cpu_vs_gpu():
    """Compare CPU vs GPU training performance"""
    print("\n=== CPU vs GPU Training Comparison ===")
    
    # Generate data
    X_train, y_train = generate_synthetic_data(2000, 32)
    X_val, y_val = generate_synthetic_data(400, 32)
    
    results = {}
    
    # CPU Training
    print("\n🔄 Training on CPU...")
    with tf.device('/CPU:0'):
        model_cpu = create_model()
        model_cpu.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        start_time = time.time()
        history_cpu = model_cpu.fit(
            X_train, y_train,
            batch_size=32,
            epochs=3,
            validation_data=(X_val, y_val),
            verbose=1
        )
        cpu_time = time.time() - start_time
        results['cpu'] = cpu_time
        print(f"CPU training completed in {cpu_time:.2f}s")
    
    # GPU Training (if available)
    gpu_devices = tf.config.list_physical_devices('GPU')
    if gpu_devices:
        print("\n🔄 Training on GPU...")
        with tf.device('/GPU:0'):
            model_gpu = create_model()
            model_gpu.compile(
                optimizer='adam',
                loss='categorical_crossentropy',
                metrics=['accuracy']
            )
            
            start_time = time.time()
            history_gpu = model_gpu.fit(
                X_train, y_train,
                batch_size=32,
                epochs=3,
                validation_data=(X_val, y_val),
                verbose=1
            )
            gpu_time = time.time() - start_time
            results['gpu'] = gpu_time
            print(f"GPU training completed in {gpu_time:.2f}s")
            
            if 'cpu' in results:
                speedup = results['cpu'] / results['gpu']
                print(f"🚀 GPU speedup: {speedup:.2f}x")
    
    return results

def mixed_precision_example():
    """Demonstrate mixed precision training"""
    print("\n=== Mixed Precision Training Example ===")
    
    # Enable mixed precision
    policy = tf.keras.mixed_precision.Policy('mixed_float16')
    tf.keras.mixed_precision.set_global_policy(policy)
    
    print(f"Mixed precision policy: {policy}")
    print(f"Compute dtype: {policy.compute_dtype}")
    print(f"Variable dtype: {policy.variable_dtype}")
    
    # Create model with mixed precision
    model = create_model()
    
    # Add loss scaling for stability
    optimizer = tf.keras.optimizers.Adam()
    optimizer = tf.keras.mixed_precision.LossScaleOptimizer(optimizer)
    
    model.compile(
        optimizer=optimizer,
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Generate data
    X_train, y_train = generate_synthetic_data(1000, 32)
    
    # Train with mixed precision
    gpu_devices = tf.config.list_physical_devices('GPU')
    if gpu_devices:
        with tf.device('/GPU:0'):
            print("🔄 Training with mixed precision...")
            start_time = time.time()
            history = model.fit(
                X_train, y_train,
                batch_size=64,  # Larger batch size for mixed precision
                epochs=2,
                verbose=1
            )
            mixed_precision_time = time.time() - start_time
            print(f"Mixed precision training completed in {mixed_precision_time:.2f}s")
    
    # Reset policy
    tf.keras.mixed_precision.set_global_policy('float32')

def inference_benchmark():
    """Benchmark inference performance"""
    print("\n=== Inference Benchmark ===")
    
    model = create_model()
    model.compile(optimizer='adam', loss='categorical_crossentropy')
    
    # Generate test data
    test_data = np.random.randn(1000, 32, 32, 3).astype(np.float32)
    
    # CPU inference
    print("🔄 CPU inference...")
    with tf.device('/CPU:0'):
        start_time = time.time()
        predictions_cpu = model.predict(test_data, batch_size=100, verbose=0)
        cpu_inference_time = time.time() - start_time
        print(f"CPU inference: {cpu_inference_time:.4f}s ({cpu_inference_time*1000/len(test_data):.2f}ms per sample)")
    
    # GPU inference
    gpu_devices = tf.config.list_physical_devices('GPU')
    if gpu_devices:
        print("🔄 GPU inference...")
        with tf.device('/GPU:0'):
            start_time = time.time()
            predictions_gpu = model.predict(test_data, batch_size=100, verbose=0)
            gpu_inference_time = time.time() - start_time
            print(f"GPU inference: {gpu_inference_time:.4f}s ({gpu_inference_time*1000/len(test_data):.2f}ms per sample)")
            
            speedup = cpu_inference_time / gpu_inference_time
            print(f"🚀 GPU inference speedup: {speedup:.2f}x")

def memory_growth_example():
    """Demonstrate GPU memory growth configuration"""
    print("\n=== GPU Memory Growth Example ===")
    
    gpu_devices = tf.config.list_physical_devices('GPU')
    if gpu_devices:
        try:
            # Enable memory growth to prevent TensorFlow from allocating all GPU memory
            for gpu in gpu_devices:
                tf.config.experimental.set_memory_growth(gpu, True)
            print("✅ GPU memory growth enabled")
            
            # Test memory allocation
            print("🔄 Testing memory allocation...")
            with tf.device('/GPU:0'):
                # Gradually allocate larger tensors
                for size in [1000, 2000, 3000, 4000]:
                    tensor = tf.random.normal([size, size])
                    result = tf.matmul(tensor, tensor)
                    print(f"Allocated and computed {size}x{size} matrix")
                    del tensor, result
            
        except RuntimeError as e:
            print(f"Memory growth configuration failed: {e}")
            print("Note: Memory growth must be set before any operations")

def main():
    """Run all TensorFlow examples"""
    print("TensorFlow Metal Examples for Mac GPU")
    print("=" * 50)
    
    # Setup TensorFlow
    gpu_available = setup_tensorflow()
    
    if not gpu_available:
        print("⚠️  GPU not available, some examples will be skipped")
    
    # Configure memory growth
    memory_growth_example()
    
    # Run examples
    train_cpu_vs_gpu()
    mixed_precision_example()
    inference_benchmark()
    
    print("\n✅ All TensorFlow examples completed!")

if __name__ == "__main__":
    main()