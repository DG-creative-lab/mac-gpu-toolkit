# Troubleshooting Guide: Mac GPU Acceleration

*Complete solutions for common issues when setting up and using GPU acceleration on Apple Silicon Macs*

---

## 📋 Common Installation Issues

### Issue 1: "MPS backend is not available"

**Symptoms:**
```python
>>> torch.backends.mps.is_available()
False
```

**Root Causes & Solutions:**

**A) Wrong PyTorch Version**
```bash
# Check current version
python -c "import torch; print(torch.__version__)"

# If version < 1.12.0, upgrade
pip uninstall torch torchvision torchaudio
pip3 install torch torchvision torchaudio
```

**B) Intel Mac (not Apple Silicon)**
```bash
# Check if you're on Apple Silicon
uname -m
# Should return: arm64 (Apple Silicon) or x86_64 (Intel)
```
*MPS only works on Apple Silicon Macs (M1/M2/M3/M4)*

**C) macOS Version Too Old**
```bash
# Check macOS version
sw_vers
# MPS requires macOS 12.3+ (Monterey)
```

**D) PyTorch Built Without MPS Support**
```python
# Check if MPS is built into PyTorch
import torch
print(f"MPS built: {torch.backends.mps.is_built()}")
```

If False, reinstall PyTorch:
```bash
pip uninstall torch torchvision torchaudio
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

### Issue 2: "RuntimeError: MPS does not support..."

**Symptoms:**
```
RuntimeError: MPS does not support [specific operation]
```

**Solutions:**

**A) Enable MPS Fallback**
```python
import os
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'

# Or in your code
try:
    result = operation_on_mps(tensor)
except RuntimeError as e:
    if "MPS" in str(e):
        result = operation_on_cpu(tensor.cpu()).to('mps')
    else:
        raise
```

**B) Use Alternative Operations**

Common problematic operations and alternatives:

Instead of: torch.linalg.svd (not supported)
Use: torch.svd (supported)

Instead of: torch.fft.fft (limited support)
Use: CPU fallback for FFT operations

Instead of: certain indexing operations
Use: torch.gather or torch.index_select


**C) Mixed Precision Issues**
```python
# If float16 causes issues, try bfloat16
model = model.to(dtype=torch.bfloat16)

# Or disable mixed precision for problematic layers
for name, module in model.named_modules():
    if isinstance(module, ProblematicLayer):
        module.to(dtype=torch.float32)
```

### Issue 3: TensorFlow Metal Issues

**Symptoms:**
```python
>>> tf.config.list_physical_devices('GPU')
[]
```

**Solutions:**

**A) Correct Installation Order**
```bash
# Uninstall all TensorFlow packages
pip uninstall tensorflow tensorflow-macos tensorflow-metal

# Install in correct order
pip install tensorflow-macos
pip install tensorflow-metal
```

**B) Python Version Compatibility**
```bash
# Check Python version
python --version
# TensorFlow Metal requires Python 3.8-3.11
```

**C) Environment Variables**
```bash
# Add to your shell profile (.zshrc or .bash_profile)
export TF_ENABLE_ONEDNN_OPTS=0
export TF_CPP_MIN_LOG_LEVEL=1
```

---

## 🔧 Runtime Issues

### Issue 4: Memory Errors

**Symptoms:**

RuntimeError: MPS backend out of memory

**Solutions:**

**A) Immediate Memory Cleanup**
```python
# Clear all caches
import gc
gc.collect()

if torch.backends.mps.is_available():
    torch.mps.empty_cache()
    
```

**B) Reduce Batch Size**
```python
# Use our adaptive batch processor
from src.optimization import AdaptiveBatchProcessor

processor = AdaptiveBatchProcessor(initial_batch_size=16)
# It will automatically adjust batch size based on memory
```

**C) Enable Gradient Checkpointing**
```python
# For training - trades compute for memory
model.gradient_checkpointing_enable()
```

**D) Use Memory-Efficient Loading**
```python
# Load model with low memory usage
model = AutoModel.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    low_cpu_mem_usage=True,
    device_map="auto"
)
```

### Issue 5: Slow Performance on MPS

**Symptoms:**
- MPS slower than expected
- No speedup over CPU

**Solutions:**

**A) Enable Mixed Precision**
```python
# Mixed precision dramatically improves MPS performance
model = model.half()  # Convert to float16

# Or use autocast
with torch.autocast(device_type='mps', dtype=torch.float16):
    output = model(input)
```

**B) Optimize Memory Access Patterns**
```python
# Ensure contiguous memory layout
tensor = tensor.contiguous()

# Use non_blocking transfers where possible
tensor = tensor.to('mps', non_blocking=True)
```

**C) Batch Operations**
```python
# Instead of processing one item at a time
for item in items:
    result = model(item.to('mps'))

# Process in batches
for batch in batched(items, batch_size=32):
    batch_tensor = torch.stack(batch).to('mps')
    results = model(batch_tensor)
```

**D) Check for CPU-GPU Transfers**
```python
# Avoid unnecessary transfers
# Bad:
for i in range(len(data)):
    item = data[i].to('mps')  # Transfer each item separately
    
# Good:
data_mps = data.to('mps')  # Transfer entire batch once
for i in range(len(data_mps)):
    item = data_mps[i]
```

### Issue 6: Model Loading Issues

**Symptoms:**
```
OSError: Unable to load weights
AttributeError: 'NoneType' object has no attribute...
```

**Solutions:**

**A) Checkpoint Compatibility**
```python
# Load checkpoint with map_location
checkpoint = torch.load('model.pth', map_location='cpu')
model.load_state_dict(checkpoint, strict=False)
model = model.to('mps')
```

**B) Mixed Architecture Models**
```python
# Some models have layers that don't support MPS
def selective_device_placement(model, primary_device='mps'):
    for name, module in model.named_modules():
        if 'embedding' in name.lower():
            module.to(primary_device)
        elif 'classifier' in name.lower():
            module.to('cpu')  # If classifier has issues on MPS
    return model
```

---

## 🐛 Framework-Specific Issues

### PyTorch Issues

**Issue: torch.compile() Not Working**
```python
# torch.compile is not stable on MPS yet
# Use manual optimizations instead
model.eval()
model = model.half()
# Skip torch.compile for now on MPS
```

**Issue: DataLoader Hanging**
```python
# On Mac, sometimes num_workers > 0 causes issues
dataloader = DataLoader(
    dataset, 
    batch_size=32,
    num_workers=0,  # Try with 0 first
    pin_memory=False  # Not needed on unified memory
)
```

### TensorFlow Issues

**Issue: Metal Device Not Found**
```python
# Force TensorFlow to see Metal device
import tensorflow as tf

# Explicit device placement
with tf.device('/GPU:0'):
    # Your TensorFlow code here
    pass

# Or configure memory growth
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    tf.config.experimental.set_memory_growth(gpus[0], True)
```

### Transformers/Hugging Face Issues

**Issue: Model Not Moving to MPS**
```python
# Explicitly move all components
model = AutoModel.from_pretrained(model_name)
model = model.to('mps')

# For models with multiple components
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    device_map="mps"  # Automatic device mapping
)
```

---

## 🔍 Debugging Tools

### Memory Debugging

```python
def debug_memory_usage():
    """Debug memory usage across all devices"""
    import psutil
    
    print("=== MEMORY DEBUG ===")
    
    # System memory
    vm = psutil.virtual_memory()
    print(f"System Memory: {vm.used/1024**3:.1f}GB / {vm.total/1024**3:.1f}GB ({vm.percent:.1f}%)")
    
    # PyTorch memory
    if torch.cuda.is_available():
        print(f"CUDA Memory: {torch.cuda.memory_allocated()/1024**3:.1f}GB allocated")
        print(f"CUDA Cached: {torch.cuda.memory_reserved()/1024**3:.1f}GB cached")
        
    # MPS doesn't have separate memory tracking
    print("===================")
```

### Performance Debugging

```python
def debug_performance_bottlenecks(model, sample_input, iterations=10):
    """Identify performance bottlenecks"""
    import time
    
    device = sample_input.device
    model = model.to(device)
    
    # Warmup
    with torch.no_grad():
        for _ in range(3):
            _ = model(sample_input)
            
    # Profile individual components
    times = {}
    
    with torch.no_grad():
        # Full forward pass
        start = time.time()
        for _ in range(iterations):
            output = model(sample_input)
            if device.type == 'mps':
                torch.mps.synchronize()
        times['full_forward'] = (time.time() - start) / iterations
        
        # Profile by layer (if possible)
        if hasattr(model, 'named_modules'):
            for name, module in model.named_modules():
                if len(list(module.children())) == 0:  # Leaf module
                    start = time.time()
                    for _ in range(iterations):
                        _ = module(sample_input)
                        if device.type == 'mps':
                            torch.mps.synchronize()
                    times[f'layer_{name}'] = (time.time() - start) / iterations
    
    # Sort by time
    sorted_times = sorted(times.items(), key=lambda x: x[1], reverse=True)
    
    print("=== PERFORMANCE BOTTLENECKS ===")
    for name, avg_time in sorted_times[:10]:  # Top 10
        print(f"{name}: {avg_time*1000:.2f}ms")
    print("===============================")
```

---

## ⚡ Performance Issues

### Issue 7: Unexpected Slowdowns

**Check 1: Thermal Throttling**
```bash
# Monitor CPU temperature
sudo powermetrics -i 1000 -n 1 --samplers smc | grep -i temp

# Check system activity
Activity Monitor > CPU tab > Look for high CPU usage
```

**Check 2: Background Processes**
```bash
# Check for memory pressure
memory_pressure
# Should show: "System-wide memory free percentage: XX%"
```

**Check 3: Power Management**
```bash
# Ensure you're plugged in for maximum performance
pmset -g ps
# Check power adapter status
```

### Issue 8: Inconsistent Performance

**Solution: Stabilize Performance**
```python
def stabilize_performance():
    """Stabilize Mac performance for benchmarking"""
    import os
    
    # Disable dynamic performance scaling
    os.system("sudo pmset -a disablesleep 1")  # Disable sleep
    
    # Set high performance mode
    os.system("sudo pmset -a lowpowermode 0")  # Disable low power mode
    
    # Warm up the GPU
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    x = torch.randn(1000, 1000, device=device)
    for _ in range(10):
        torch.matmul(x, x)
        if device.type == 'mps':
            torch.mps.synchronize()
```

---

## 🔧 Environment Issues

### Issue 9: Conflicting Python Environments

**Symptoms:**
- Import errors
- Mixed package versions
- Inconsistent behavior

**Solution: Clean Environment Setup**
```bash
# Remove all existing ML packages
pip uninstall torch torchvision torchaudio tensorflow tensorflow-macos tensorflow-metal

# Create fresh conda environment
conda create -n gpu_mac python=3.11
conda activate gpu_mac

# Install packages in correct order
pip3 install torch torchvision torchaudio
pip install tensorflow-macos tensorflow-metal
pip install -r requirements.txt
```

### Issue 10: Homebrew Conflicts

**Symptoms:**
- Segmentation faults
- Library loading errors

**Solution:**
```bash
# Check for conflicting libraries
brew list | grep -E "(python|numpy|opencv)"

# Remove conflicting packages
brew uninstall --ignore-dependencies opencv numpy

# Use pip-managed packages instead
pip install opencv-python numpy
```

---

## 🏷️ Model-Specific Issues

### Issue 11: Large Language Model Problems

**A) Model Too Large for Memory**
```python
# Use 8-bit or 4-bit quantization
from transformers import BitsAndBytesConfig

quantization_config = BitsAndBytesConfig(
    load_in_8bit=True,  # or load_in_4bit=True
    bnb_4bit_compute_dtype=torch.float16
)

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=quantization_config,
    device_map="auto"
)
```

**B) Slow Text Generation**
```python
# Optimize generation parameters
generation_config = {
    'do_sample': True,
    'temperature': 0.7,
    'top_p': 0.9,
    'pad_token_id': tokenizer.eos_token_id,
    'use_cache': True,  # Enable KV caching
    'max_new_tokens': 512
}

# Use optimized generation
with torch.autocast(device_type='mps', dtype=torch.float16):
    outputs = model.generate(**generation_config)
```

### Issue 12: Computer Vision Model Issues

**A) Image Preprocessing Errors**
```python
# Ensure proper tensor format and device placement
from torchvision import transforms

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                        std=[0.229, 0.224, 0.225])
])

# Apply transform then move to device
image_tensor = transform(image).unsqueeze(0).to('mps')
```

**B) ONNX Model Conversion Issues**
```python
# Export with explicit opset version for Mac compatibility
torch.onnx.export(
    model,
    dummy_input,
    "model.onnx",
    opset_version=11,  # Use compatible opset
    do_constant_folding=True,
    input_names=['input'],
    output_names=['output']
)
```

---

## 🚀 Performance Troubleshooting

### Issue 13: Slower Than Expected Performance

**Diagnosis Checklist:**

1. **Check Device Usage**
```python
# Verify you're actually using the GPU
print(f"Model device: {next(model.parameters()).device}")
print(f"Input device: {input_tensor.device}")
```

2. **Profile Memory Transfers**
```python
# Look for expensive CPU-GPU transfers
with torch.profiler.profile(
    activities=[torch.profiler.ProfilerActivity.CPU],
    record_shapes=True
) as prof:
    output = model(input)

print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=10))
```

3. **Check Batch Size**
```python
# Use our batch size optimizer
from src.optimization import BatchSizeOptimizer

optimizer = BatchSizeOptimizer()
optimal_batch = optimizer.find_optimal_batch_size(model, sample_input)
print(f"Optimal batch size: {optimal_batch}")
```

### Issue 14: Memory Leaks

**Symptoms:**
- Memory usage keeps increasing
- Eventually runs out of memory
- Performance degrades over time

**Solutions:**

```python
# A) Explicit memory management
def process_with_cleanup(data_batch):
    try:
        result = model(data_batch)
        return result.cpu()  # Move result to CPU immediately
    finally:
        del data_batch
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
        gc.collect()

# B) Use context managers
from src.optimization import memory_efficient_inference

with memory_efficient_inference():
    for batch in dataloader:
        result = model(batch)
        # Automatic cleanup happens here
```

---

## 🛠️ Advanced Debugging

### Debug Script Template

```python
#!/usr/bin/env python3
"""
Advanced debugging script for Mac GPU issues
"""

import torch
import platform
import sys
import subprocess
import psutil

def comprehensive_debug():
    print("🔍 COMPREHENSIVE MAC GPU DEBUG REPORT")
    print("=" * 60)
    
    # System information
    print(f"🖥️  System: {platform.system()} {platform.release()}")
    print(f"🏗️  Architecture: {platform.machine()}")
    print(f"🐍 Python: {sys.version}")
    
    # Hardware information
    try:
        result = subprocess.run(['sysctl', '-n', 'machdep.cpu.brand_string'], 
                              capture_output=True, text=True)
        print(f"💻 CPU: {result.stdout.strip()}")
    except:
        print("💻 CPU: Unable to detect")
    
    # Memory information
    memory = psutil.virtual_memory()
    print(f"💾 Memory: {memory.total/1024**3:.1f}GB total, {memory.available/1024**3:.1f}GB available")
    
    # PyTorch information
    print(f"\n🔥 PyTorch: {torch.__version__}")
    print(f"🧮 MPS Available: {torch.backends.mps.is_available()}")
    print(f"🔨 MPS Built: {torch.backends.mps.is_built()}")
    
    # Test MPS functionality
    if torch.backends.mps.is_available():
        try:
            x = torch.randn(100, 100, device='mps')
            y = torch.randn(100, 100, device='mps')
            z = torch.matmul(x, y)
            print("✅ MPS Basic Operations: Working")
        except Exception as e:
            print(f"❌ MPS Basic Operations: {e}")
    
    # TensorFlow information
    try:
        import tensorflow as tf
        print(f"\n🧠 TensorFlow: {tf.__version__}")
        gpus = tf.config.list_physical_devices('GPU')
        print(f"🎮 TF GPU Devices: {len(gpus)}")
        for gpu in gpus:
            print(f"   - {gpu}")
    except ImportError:
        print("\n🧠 TensorFlow: Not installed")
    
    # MLX information (if available)
    try:
        import mlx.core as mx
        print(f"\n🍎 MLX Available: {mx.metal.is_available()}")
    except ImportError:
        print("\n🍎 MLX: Not installed")
    
    print("=" * 60)

if __name__ == "__main__":
    comprehensive_debug()
```

---

### Community Resources

- **PyTorch MPS Issues**: [PyTorch GitHub Issues](https://github.com/pytorch/pytorch/issues)
- **TensorFlow Metal**: [TensorFlow Metal Issues](https://github.com/tensorflow/metal/issues)
- **Apple Developer Forums**: [Machine Learning](https://developer.apple.com/forums/topics/machine-learning)

### Emergency Fallback

If nothing works, you can always fall back to CPU:

```python
# Force CPU usage
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '0'

device = torch.device('cpu')
model = model.to(device)
```

---

## 🧪 Testing Your Fix

After applying any solution, run our verification script:

```bash
python scripts/verify_gpu.py
```

This will confirm that:
- ✅ GPU acceleration is working
- ✅ Memory management is optimal  
- ✅ Performance is as expected
- ✅ No errors in basic operations
