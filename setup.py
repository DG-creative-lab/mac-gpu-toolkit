import subprocess
import sys
import os
from pathlib import Path

class Colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    END = '\033[0m'
    BOLD = '\033[1m'

def print_header(text):
    print(f"\n{Colors.HEADER}{Colors.BOLD}{'='*60}{Colors.END}")
    print(f"{Colors.HEADER}{Colors.BOLD}{text.center(60)}{Colors.END}")
    print(f"{Colors.HEADER}{Colors.BOLD}{'='*60}{Colors.END}\n")

def print_success(text):
    print(f"{Colors.GREEN}✅ {text}{Colors.END}")

def print_warning(text):
    print(f"{Colors.WARNING}⚠️  {text}{Colors.END}")

def print_error(text):
    print(f"{Colors.FAIL}❌ {text}{Colors.END}")

def print_info(text):
    print(f"{Colors.CYAN}ℹ️  {text}{Colors.END}")

def run_script(script_name, description):
    """Run a script and handle output"""
    print_info(f"Running {description}...")
    try:
        result = subprocess.run([sys.executable, f"scripts/{script_name}"], 
                              capture_output=True, text=True, check=True)
        print(result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print_error(f"Failed to run {script_name}")
        print(e.stderr)
        return False

def main():
    print_header("Mac GPU Acceleration Toolkit Setup")
    
    # Check if you're on macOS
    if not sys.platform.startswith('darwin'):
        print_error("This toolkit is designed for macOS with Apple Silicon")
        return False
    
    print("This interactive setup will:")
    print("1. Check your current environment")
    print("2. Optionally clean install dependencies")
    print("3. Verify GPU acceleration is working")
    print("4. Run performance benchmarks")
    
    if input("\nProceed? (y/N): ").lower() != 'y':
        print("Setup cancelled.")
        return False
    
    # Step 1: Environment check
    print_header("Step 1: Environment Assessment")
    if not run_script("check_environment.py", "environment assessment"):
        print_warning("Environment check had issues, but continuing...")
    
    # Step 2: Dependencies (optional)
    print_header("Step 2: Dependencies Installation")
    install_deps = input("Install/update dependencies? This will create a clean environment (y/N): ")
    if install_deps.lower() == 'y':
        if not run_script("install_dependencies.py", "dependencies installation"):
            print_error("Failed to install dependencies. Please check the output above.")
            return False
    
    # Step 3: Verification
    print_header("Step 3: GPU Verification")
    if not run_script("verify_gpu.py", "GPU verification"):
        print_error("GPU verification failed. Please check your installation.")
        return False
    
    # Step 4: Benchmarks (optional)
    print_header("Step 4: Performance Benchmarks")
    run_benchmarks = input("Run performance benchmarks? This may take a few minutes (y/N): ")
    if run_benchmarks.lower() == 'y':
        run_script("benchmark_performance.py", "performance benchmarks")
    
    print_header("Setup Complete!")
    print_success("Your Mac is now ready for GPU-accelerated machine learning!")
    print_info("Check the examples/ directory for usage examples")
    print_info("Run 'python scripts/troubleshoot.py' if you encounter issues")
    
    return True

if __name__ == "__main__":
    main()