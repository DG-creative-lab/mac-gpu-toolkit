import time
import sys
import traceback

def test_pytorch_mps():
    """Test PyTorch MPS functionality"""
    print("=== PyTorch MPS Verification ===")
    
    try:
        import torch
        print(f"PyTorch version: {torch.__version__}")
        
        # Check MPS availability
        if not torch.backends.mps.is_available():
            print("❌ MPS not available")
            return False
        
        print("✅ MPS is available")
        
        if not torch.backends.mps.is_built():
            print("❌ MPS not built correctly")
            return False
        
        print("✅ MPS built correctly")
        
        # Test device creation
        device = torch.device('mps')
        print(f"✅ MPS device created: {device}")
        
        # Test tensor operations
        print("🔄 Testing tensor operations...")
        
        # Basic operations
        x = torch.randn(1000, 1000, device=device)
        y = torch.randn(1000, 1000, device=device)
        
        start_time = time.time()
        z = torch.matmul(x, y)
        torch.mps.synchronize()  # Ensure operation completes
        end_time = time.time()
        
        print(f"✅ Matrix multiplication (1000x1000): {end_time - start_time:.4f}s")
        
        # Test autograd
        x.requires_grad_(True)
        y.requires_grad_(True)
        z = torch.sum(x * y)
        z.backward()
        
        if x.grad is not None and y.grad is not None:
            print("✅ Autograd working on MPS")
        else:
            print("❌ Autograd not working on MPS")
            return False
        
        # Test mixed precision
        print("🔄 Testing mixed precision...")
        try:
            with torch.autocast(device_type='mps', dtype=torch.float16):
                x_fp16 = torch.randn(500, 500, device=device)
                y_fp16 = torch.randn(500, 500, device=device)
                z_fp16 = torch.matmul(x_fp16, y_fp16)
            print("✅ Mixed precision (float16) working")
        except Exception as e:
            print(f"⚠️  Mixed precision failed: {e}")
        
        # Memory management test
        print("🔄 Testing memory management...")
        initial_memory = torch.mps.current_allocated_memory()
        
        large_tensor = torch.randn(2000, 2000, device=device)
        after_alloc = torch.mps.current_allocated_memory()
        
        del large_tensor
        torch.mps.empty_cache()
        after_cleanup = torch.mps.current_allocated_memory()
        
        print(f"✅ Memory: Initial={initial_memory/1024/1024:.1f}MB, "
              f"After alloc={after_alloc/1024/1024:.1f}MB, "
              f"After cleanup={after_cleanup/1024/1024:.1f}MB")
        
        return True
        
    except ImportError:
        print("❌ PyTorch not installed")
        return False
    except Exception as e:
        print(f"❌ PyTorch MPS test failed: {e}")
        traceback.print_exc()
        return False

def test_tensorflow_metal():
    """Test TensorFlow Metal functionality"""
    print("\n=== TensorFlow Metal Verification ===")
    
    try:
        import tensorflow as tf
        print(f"TensorFlow version: {tf.__version__}")
        
        # Check for GPU devices
        physical_devices = tf.config.list_physical_devices('GPU')
        if not physical_devices:
            print("❌ No GPU devices found")
            return False
        
        print(f"✅ Found {len(physical_devices)} GPU device(s):")
        for i, device in enumerate(physical_devices):
            print(f"   Device {i}: {device}")
        
        # Test basic operations
        print("🔄 Testing tensor operations...")
        
        with tf.device('/GPU:0'):
            x = tf.random.normal([1000, 1000])
            y = tf.random.normal([1000, 1000])
            
            start_time = time.time()
            z = tf.matmul(x, y)
            end_time = time.time()
            
            print(f"✅ Matrix multiplication (1000x1000): {end_time - start_time:.4f}s")
        
        # Test gradient computation
        print("🔄 Testing gradients...")
        
        with tf.device('/GPU:0'):
            x = tf.Variable(tf.random.normal([100, 100]))
            
            with tf.GradientTape() as tape:
                y = tf.reduce_sum(x ** 2)
            
            grads = tape.gradient(y, x)
            
            if grads is not None:
                print("✅ Gradient computation working")
            else:
                print("❌ Gradient computation failed")
                return False
        
        # Test mixed precision
        print("🔄 Testing mixed precision...")
        try:
            policy = tf.keras.mixed_precision.Policy('mixed_float16')
            tf.keras.mixed_precision.set_global_policy(policy)
            
            with tf.device('/GPU:0'):
                x = tf.random.normal([500, 500])
                y = tf.random.normal([500, 500])
                z = tf.matmul(x, y)
            
            print("✅ Mixed precision working")
            
            # Reset policy
            tf.keras.mixed_precision.set_global_policy('float32')
            
        except Exception as e:
            print(f"⚠️  Mixed precision failed: {e}")
        
        return True
        
    except ImportError:
        print("❌ TensorFlow not installed")
        return False
    except Exception as e:
        print(f"❌ TensorFlow Metal test failed: {e}")
        traceback.print_exc()
        return False

def test_mlx():
    """Test MLX functionality"""
    print("\n=== MLX Verification ===")
    
    try:
        import mlx.core as mx
        import mlx.nn as nn
        print(f"MLX available")
        
        if not mx.metal.is_available():
            print("❌ MLX Metal not available")
            return False
        
        print("✅ MLX Metal is available")
        
        # Test basic operations
        print("🔄 Testing MLX operations...")
        
        x = mx.random.normal([1000, 1000])
        y = mx.random.normal([1000, 1000])
        
        start_time = time.time()
        z = mx.matmul(x, y)
        mx.eval(z)  # Ensure computation completes
        end_time = time.time()
        
        print(f"✅ Matrix multiplication (1000x1000): {end_time - start_time:.4f}s")
        
        # Test neural network layer
        print("🔄 Testing neural network layer...")
        
        layer = nn.Linear(100, 50)
        x = mx.random.normal([32, 100])  # batch_size=32, features=100
        y = layer(x)
        mx.eval(y)
        
        print(f"✅ Neural network layer: input {x.shape} -> output {y.shape}")
        
        return True
        
    except ImportError:
        print("❌ MLX not installed")
        return False
    except Exception as e:
        print(f"❌ MLX test failed: {e}")
        traceback.print_exc()
        return False

def performance_comparison():
    """Compare CPU vs GPU performance"""
    print("\n=== Performance Comparison ===")
    
    try:
        import torch
        
        # CPU test
        print("🔄 CPU performance test...")
        device_cpu = torch.device('cpu')
        x_cpu = torch.randn(2048, 2048, device=device_cpu)
        y_cpu = torch.randn(2048, 2048, device=device_cpu)
        
        start_time = time.time()
        z_cpu = torch.matmul(x_cpu, y_cpu)
        cpu_time = time.time() - start_time
        
        print(f"CPU: {cpu_time:.4f}s")
        
        # GPU test (if available)
        if torch.backends.mps.is_available():
            print("🔄 GPU performance test...")
            device_gpu = torch.device('mps')
            x_gpu = torch.randn(2048, 2048, device=device_gpu)
            y_gpu = torch.randn(2048, 2048, device=device_gpu)
            
            start_time = time.time()
            z_gpu = torch.matmul(x_gpu, y_gpu)
            torch.mps.synchronize()
            gpu_time = time.time() - start_time
            
            print(f"GPU: {gpu_time:.4f}s")
            
            speedup = cpu_time / gpu_time
            print(f"✅ Speedup: {speedup:.2f}x faster on GPU")
        else:
            print("⚠️  GPU not available for comparison")
        
    except Exception as e:
        print(f"❌ Performance comparison failed: {e}")

def main():
    """Run all verification tests"""
    print("Mac GPU Comprehensive Verification")
    print("=" * 50)
    
    results = {}
    
    # Run tests
    results['pytorch'] = test_pytorch_mps()
    results['tensorflow'] = test_tensorflow_metal()
    results['mlx'] = test_mlx()
    
    # Performance comparison
    performance_comparison()
    
    # Summary
    print("\n" + "=" * 50)
    print("VERIFICATION SUMMARY")
    print("=" * 50)
    
    if results['pytorch']:
        print("✅ PyTorch MPS: WORKING")
    else:
        print("❌ PyTorch MPS: FAILED")
    
    if results['tensorflow']:
        print("✅ TensorFlow Metal: WORKING")
    else:
        print("❌ TensorFlow Metal: FAILED")
    
    if results['mlx']:
        print("✅ MLX: WORKING")
    else:
        print("❌ MLX: FAILED")
    
    working_frameworks = sum(results.values())
    
    if working_frameworks == 3:
        print("\n🎉 ALL FRAMEWORKS WORKING! Your Mac is fully optimized for GPU acceleration!")
    elif working_frameworks >= 1:
        print(f"\n✅ {working_frameworks}/3 frameworks working. You're ready to start!")
    else:
        print("\n❌ No frameworks working properly. Please check your installation.")
    
    print(f"\n💡 Next steps:")
    print(f"   - Check examples/ directory for usage examples")
    print(f"   - Run benchmarks: python scripts/benchmark_performance.py")
    print(f"   - If issues persist: python scripts/troubleshoot.py")

if __name__ == "__main__":
    main()