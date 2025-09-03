import subprocess
import sys
import os
import venv
from pathlib import Path

def run_command(cmd, description):
    """Run a command and handle output"""
    print(f"🔄 {description}...")
    try:
        subprocess.run(cmd, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {description} completed")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed")
        print(f"Error: {e.stderr}")
        return False

def create_virtual_environment():
    """Create a clean virtual environment"""
    env_path = Path("venv_mac_gpu")
    
    if env_path.exists():
        print("🗑️  Removing existing virtual environment...")
        import shutil
        shutil.rmtree(env_path)
    
    print("🔧 Creating new virtual environment...")
    venv.create(env_path, with_pip=True)
    
    # Get the python executable path
    if sys.platform == "win32":
        python_exe = env_path / "Scripts" / "python.exe"
        pip_exe = env_path / "Scripts" / "pip.exe"
    else:
        python_exe = env_path / "bin" / "python"
        pip_exe = env_path / "bin" / "pip"
    
    print(f"✅ Virtual environment created at {env_path}")
    print(f"💡 To activate: source {env_path}/bin/activate")
    
    return str(python_exe), str(pip_exe)

def install_pytorch(pip_exe):
    """Install PyTorch with MPS support"""
    print("\n=== Installing PyTorch ===")
    
    # Uninstall existing pytorch
    run_command(f"{pip_exe} uninstall -y torch torchvision torchaudio", 
               "Removing existing PyTorch")
    
    # Install latest PyTorch with MPS support
    cmd = f"{pip_exe} install torch torchvision torchaudio"
    return run_command(cmd, "Installing PyTorch with MPS support")

def install_tensorflow(pip_exe):
    """Install TensorFlow with Metal support"""
    print("\n=== Installing TensorFlow ===")
    
    # Uninstall existing tensorflow
    run_command(f"{pip_exe} uninstall -y tensorflow tensorflow-metal tensorflow-macos", 
               "Removing existing TensorFlow")
    
    # Install TensorFlow for macOS with Metal support
    success = True
    success &= run_command(f"{pip_exe} install tensorflow-macos", 
                          "Installing TensorFlow for macOS")
    success &= run_command(f"{pip_exe} install tensorflow-metal", 
                          "Installing TensorFlow Metal plugin")
    
    return success

def install_mlx(pip_exe):
    """Install MLX (Apple's native ML framework)"""
    print("\n=== Installing MLX ===")
    
    return run_command(f"{pip_exe} install mlx", "Installing MLX")

def install_common_packages(pip_exe):
    """Install commonly used ML packages"""
    print("\n=== Installing Common ML Packages ===")
    
    packages = [
        "numpy",
        "pandas",
        "scikit-learn",
        "matplotlib",
        "jupyter",
        "tqdm",
        "pillow",
        "requests"
    ]
    
    success = True
    for package in packages:
        success &= run_command(f"{pip_exe} install {package}", 
                              f"Installing {package}")
    
    return success

def install_optional_packages(pip_exe):
    """Install optional but useful packages"""
    print("\n=== Installing Optional Packages ===")
    
    optional_packages = {
        "transformers": "Hugging Face Transformers",
        "datasets": "Hugging Face Datasets", 
        "accelerate": "Hugging Face Accelerate",
        "marker-pdf": "Marker PDF processing",
        "streamlit": "Streamlit for web apps"
    }
    
    for package, description in optional_packages.items():
        install = input(f"Install {description}? ({package}) (y/N): ")
        if install.lower() == 'y':
            run_command(f"{pip_exe} install {package}", f"Installing {package}")

def main():
    """Main installation process"""
    print("Mac GPU Dependencies Installation")
    print("=" * 40)
    
    # Check if we're on macOS
    if not sys.platform.startswith('darwin'):
        print("❌ This script is designed for macOS")
        return False
    
    print("This will:")
    print("1. Create a clean virtual environment")
    print("2. Install PyTorch with MPS support")
    print("3. Install TensorFlow with Metal support")
    print("4. Install MLX (Apple's native framework)")
    print("5. Install common ML packages")
    print("6. Optionally install additional packages")
    
    if input("\nProceed? (y/N): ").lower() != 'y':
        print("Installation cancelled.")
        return False
    
    # Create virtual environment
    python_exe, pip_exe = create_virtual_environment()
    
    # Upgrade pip
    run_command(f"{pip_exe} install --upgrade pip setuptools wheel", 
               "Upgrading pip and tools")
    
    # Install core packages
    success = True
    success &= install_pytorch(pip_exe)
    success &= install_tensorflow(pip_exe)
    success &= install_mlx(pip_exe)
    success &= install_common_packages(pip_exe)
    
    # Optional packages
    install_optional = input("\nInstall optional packages? (y/N): ")
    if install_optional.lower() == 'y':
        install_optional_packages(pip_exe)
    
    # Create requirements.txt
    print("\n🔄 Generating requirements.txt...")
    run_command(f"{pip_exe} freeze > requirements.txt", 
               "Generating requirements.txt")
    
    # Summary
    print("\n" + "=" * 50)
    if success:
        print("🎉 Installation completed successfully!")
        print(f"\n📁 Virtual environment: venv_mac_gpu/")
        print(f"🐍 Python executable: {python_exe}")
        print(f"📦 Pip executable: {pip_exe}")
        print(f"📋 Requirements saved to: requirements.txt")
        
        print("\n🚀 Next steps:")
        print("1. Activate environment: source venv_mac_gpu/bin/activate")
        print("2. Run verification: python scripts/verify_gpu.py")
        print("3. Run benchmarks: python scripts/benchmark_performance.py")
    else:
        print("⚠️  Installation completed with some issues")
        print("Check the output above for details")
    
    return success

if __name__ == "__main__":
    main()