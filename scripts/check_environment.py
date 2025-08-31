import sys
import platform
import subprocess
import importlib.util

def check_system():
    """Check system information"""
    print("=== System Information ===")
    print(f"Platform: {platform.platform()}")
    print(f"Architecture: {platform.machine()}")
    print(f"Python version: {sys.version}")
    print(f"Python executable: {sys.executable}")
    
    # Check if Apple Silicon
    is_apple_silicon = platform.machine() == 'arm64'
    if is_apple_silicon:
        print("✅ Apple Silicon detected")
    else:
        print("⚠️  Not Apple Silicon - GPU acceleration may not work optimally")
    
    return is_apple_silicon

def check_package(package_name, import_name=None, version_attr='__version__'):
    """Check if a package is installed and get version"""
    if import_name is None:
        import_name = package_name
    
    try:
        spec = importlib.util.find_spec(import_name)
        if spec is None:
            return False, None
        
        module = importlib.import_module(import_name)
        version = getattr(module, version_attr, 'Unknown')
        return True, version
    except ImportError:
        return False, None

def check_pytorch():
    """Check PyTorch installation and MPS support"""
    print("\n=== PyTorch Status ===")
    
    installed, version = check_package('torch')
    if not installed:
        print("❌ PyTorch not installed")
        return False
    
    print(f"✅ PyTorch {version} installed")
    
    try:
        import torch
        
        # Check MPS availability
        if torch.backends.mps.is_available():
            print("✅ MPS backend available")
            if torch.backends.mps.is_built():
                print("✅ MPS backend built correctly")
            else:
                print("⚠️  MPS backend not built correctly")
        else:
            print("❌ MPS backend not available")
        
        # Test basic MPS operation
        try:
            device = torch.device('mps')
            x = torch.randn(10, 10, device=device)
            y = x * 2
            print("✅ Basic MPS operations working")
            return True
        except Exception as e:
            print(f"❌ MPS operations failed: {e}")
            return False
            
    except ImportError as e:
        print(f"❌ Error importing torch: {e}")
        return False

def check_tensorflow():
    """Check TensorFlow installation and Metal support"""
    print("\n=== TensorFlow Status ===")
    
    installed, version = check_package('tensorflow')
    if not installed:
        print("❌ TensorFlow not installed")
        return False
    
    print(f"✅ TensorFlow {version} installed")
    
    try:
        import tensorflow as tf
        
        # Check for GPU devices
        physical_devices = tf.config.list_physical_devices('GPU')
        if physical_devices:
            print(f"✅ GPU devices found: {len(physical_devices)}")
            for i, device in enumerate(physical_devices):
                print(f"   Device {i}: {device}")
            return True
        else:
            print("❌ No GPU devices found")
            
            # Check if tensorflow-metal is installed
            metal_installed, _ = check_package('tensorflow-metal')
            if not metal_installed:
                print("💡 Try installing tensorflow-metal: pip install tensorflow-metal")
            
            return False
            
    except ImportError as e:
        print(f"❌ Error importing tensorflow: {e}")
        return False

def check_mlx():
    """Check MLX installation"""
    print("\n=== MLX Status ===")
    
    installed, version = check_package('mlx')
    if not installed:
        print("❌ MLX not installed")
        print("💡 MLX is Apple's native ML framework. Install with: pip install mlx")
        return False
    
    print(f"✅ MLX {version} installed")
    
    try:
        import mlx.core as mx
        if mx.metal.is_available():
            print("✅ MLX Metal support available")
            return True
        else:
            print("❌ MLX Metal support not available")
            return False
    except ImportError as e:
        print(f"❌ Error importing mlx: {e}")
        return False

def check_common_packages():
    """Check other commonly used ML packages"""
    print("\n=== Other ML Packages ===")
    
    packages = [
        'numpy',
        'pandas',
        'scikit-learn',
        'matplotlib',
        'jupyter',
        'transformers',
        'datasets'
    ]
    
    for package in packages:
        installed, version = check_package(package)
        if installed:
            print(f"✅ {package} {version}")
        else:
            print(f"❌ {package} not installed")

def main():
    """Main environment check"""
    print("Mac GPU Environment Assessment")
    print("=" * 40)
    
    # System check
    is_apple_silicon = check_system()
    
    # Package checks
    pytorch_ok = check_pytorch()
    tensorflow_ok = check_tensorflow()
    mlx_ok = check_mlx()
    
    # Other packages
    check_common_packages()
    
    # Summary
    print("\n=== Summary ===")
    if is_apple_silicon:
        print("✅ System: Apple Silicon Mac")
    else:
        print("⚠️  System: Not Apple Silicon")
    
    if pytorch_ok:
        print("✅ PyTorch: Ready for GPU acceleration")
    else:
        print("❌ PyTorch: Issues detected")
    
    if tensorflow_ok:
        print("✅ TensorFlow: Ready for GPU acceleration")
    else:
        print("❌ TensorFlow: Issues detected")
    
    if mlx_ok:
        print("✅ MLX: Available for native optimization")
    else:
        print("❌ MLX: Not available")
    
    if pytorch_ok or tensorflow_ok:
        print("\n🎉 At least one GPU framework is working!")
    else:
        print("\n⚠️  No GPU frameworks are properly configured")
        print("💡 Run 'python scripts/install_dependencies.py' to set up")

if __name__ == "__main__":
    main()
