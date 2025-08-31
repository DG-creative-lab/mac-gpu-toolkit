import platform
import psutil
import torch
import logging
from typing import Optional, Dict, Any, List
from dataclasses import dataclass
from enum import Enum

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DeviceType(Enum):
    """Supported device types"""
    MPS = "mps"
    CUDA = "cuda"
    CPU = "cpu"


@dataclass
class DeviceInfo:
    """Device information container"""
    device_type: DeviceType
    device: torch.device
    memory_gb: float
    compute_capability: Optional[str] = None
    is_available: bool = True
    performance_score: float = 0.0


class DeviceManager:
    """
    Intelligent device selection and management for Mac and cross-platform ML workloads
    """
    
    def __init__(self, preferred_device: Optional[str] = None, fallback_enabled: bool = True):
        """
        Initialize the device manager
        
        Args:
            preferred_device: Force specific device ('mps', 'cuda', 'cpu')
            fallback_enabled: Whether to fallback to CPU if GPU unavailable
        """
        self.preferred_device = preferred_device
        self.fallback_enabled = fallback_enabled
        self.device_info = self._detect_devices()
        self.current_device = self._select_optimal_device()
        
    def _detect_devices(self) -> Dict[str, DeviceInfo]:
        """Detect and analyze available devices"""
        devices = {}
        
        # Check MPS (Apple Silicon)
        if self._is_mps_available():
            devices['mps'] = self._get_mps_info()
            
        # Check CUDA
        if self._is_cuda_available():
            devices['cuda'] = self._get_cuda_info()
            
        # CPU is always available
        devices['cpu'] = self._get_cpu_info()
        
        return devices
    
    def _is_mps_available(self) -> bool:
        """Check if MPS (Metal Performance Shaders) is available"""
        try:
            return (
                platform.system() == 'Darwin' and
                torch.backends.mps.is_available() and
                torch.backends.mps.is_built()
            )
        except Exception as e:
            logger.warning(f"Error checking MPS availability: {e}")
            return False
    
    def _is_cuda_available(self) -> bool:
        """Check if CUDA is available"""
        try:
            return torch.cuda.is_available()
        except Exception as e:
            logger.warning(f"Error checking CUDA availability: {e}")
            return False
    
    def _get_mps_info(self) -> DeviceInfo:
        """Get MPS device information"""
        # Get system memory (unified on Apple Silicon)
        memory_gb = psutil.virtual_memory().total / (1024**3)
        
        # Estimate GPU performance score based on chip
        performance_score = self._estimate_apple_silicon_performance()
        
        return DeviceInfo(
            device_type=DeviceType.MPS,
            device=torch.device('mps'),
            memory_gb=memory_gb,
            compute_capability=self._get_apple_silicon_chip(),
            performance_score=performance_score
        )
    
    def _get_cuda_info(self) -> DeviceInfo:
        """Get CUDA device information"""
        device = torch.device('cuda')
        properties = torch.cuda.get_device_properties(0)
        memory_gb = properties.total_memory / (1024**3)
        
        return DeviceInfo(
            device_type=DeviceType.CUDA,
            device=device,
            memory_gb=memory_gb,
            compute_capability=f"{properties.major}.{properties.minor}",
            performance_score=self._estimate_cuda_performance(properties)
        )
    
    def _get_cpu_info(self) -> DeviceInfo:
        """Get CPU device information"""
        memory_gb = psutil.virtual_memory().total / (1024**3)
        
        return DeviceInfo(
            device_type=DeviceType.CPU,
            device=torch.device('cpu'),
            memory_gb=memory_gb,
            compute_capability=platform.processor(),
            performance_score=1.0  # Base score
        )
    
    def _get_apple_silicon_chip(self) -> str:
        """Detect Apple Silicon chip type"""
        try:
            import subprocess
            result = subprocess.run(['sysctl', '-n', 'machdep.cpu.brand_string'], 
                                 capture_output=True, text=True)
            chip_info = result.stdout.strip()
            
            if 'M4' in chip_info:
                return 'M4'
            elif 'M3' in chip_info:
                return 'M3'
            elif 'M2' in chip_info:
                return 'M2'
            elif 'M1' in chip_info:
                return 'M1'
            else:
                return 'Apple Silicon'
        except Exception:
            return 'Unknown Apple Silicon'
    
    def _estimate_apple_silicon_performance(self) -> float:
        """Estimate relative performance of Apple Silicon chips"""
        chip = self._get_apple_silicon_chip()
        
        # Performance estimates relative to M1 base
        performance_map = {
            'M4': 7.5,   # M4 Pro/Max
            'M3': 6.0,   # M3 Pro/Max  
            'M2': 4.5,   # M2 Pro/Max
            'M1': 3.0,   # M1 Pro/Max
        }
        
        # Adjust for Pro/Max variants
        try:
            import subprocess
            result = subprocess.run(['system_profiler', 'SPHardwareDataType'], 
                                 capture_output=True, text=True)
            if 'Pro' in result.stdout or 'Max' in result.stdout:
                base_score = performance_map.get(chip, 3.0)
                return base_score * 1.3  # Pro/Max bonus
        except Exception:
            pass
            
        return performance_map.get(chip, 3.0)
    
    def _estimate_cuda_performance(self, properties) -> float:
        """Estimate CUDA GPU performance"""
        # Simple heuristic based on compute capability and memory
        cc_score = properties.major + (properties.minor / 10)
        memory_score = min(properties.total_memory / (1024**3) / 8, 2.0)  # Cap at 2x
        return cc_score * memory_score
    
    def _select_optimal_device(self) -> torch.device:
        """Select the optimal device based on availability and performance"""
        if self.preferred_device:
            if self.preferred_device in self.device_info:
                return self.device_info[self.preferred_device].device
            else:
                logger.warning(f"Preferred device '{self.preferred_device}' not available")
        
        # Select based on performance score
        available_devices = [info for info in self.device_info.values() if info.is_available]
        
        if not available_devices:
            raise RuntimeError("No available compute devices found")
        
        # Sort by performance score, prefer GPU over CPU
        optimal = max(available_devices, 
                     key=lambda d: (d.device_type != DeviceType.CPU, d.performance_score))
        
        logger.info(f"Selected device: {optimal.device_type.value} "
                   f"(performance score: {optimal.performance_score:.1f})")
        
        return optimal.device
    
    def get_device(self) -> torch.device:
        """Get the current optimal device"""
        return self.current_device
    
    def get_device_info(self, device_type: Optional[str] = None) -> DeviceInfo:
        """Get information about a specific device"""
        if device_type is None:
            device_type = self.current_device.type
            
        return self.device_info.get(device_type)
    
    def switch_device(self, device_type: str) -> bool:
        """
        Switch to a different device
        
        Args:
            device_type: Target device type ('mps', 'cuda', 'cpu')
            
        Returns:
            bool: True if switch successful
        """
        if device_type not in self.device_info:
            logger.error(f"Device type '{device_type}' not available")
            return False
            
        self.current_device = self.device_info[device_type].device
        logger.info(f"Switched to device: {device_type}")
        return True
    
    def get_memory_info(self) -> Dict[str, float]:
        """Get memory information for current device"""
        device_type = self.current_device.type
        
        if device_type == 'mps':
            # For MPS, we use system memory
            memory = psutil.virtual_memory()
            return {
                'total_gb': memory.total / (1024**3),
                'available_gb': memory.available / (1024**3),
                'used_gb': memory.used / (1024**3),
                'percent_used': memory.percent
            }
        elif device_type == 'cuda':
            # For CUDA, use GPU memory
            total = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            allocated = torch.cuda.memory_allocated(0) / (1024**3)
            cached = torch.cuda.memory_reserved(0) / (1024**3)
            return {
                'total_gb': total,
                'allocated_gb': allocated,
                'cached_gb': cached,
                'free_gb': total - cached,
                'percent_used': (cached / total) * 100
            }
        else:
            # CPU memory
            memory = psutil.virtual_memory()
            return {
                'total_gb': memory.total / (1024**3),
                'available_gb': memory.available / (1024**3),
                'used_gb': memory.used / (1024**3),
                'percent_used': memory.percent
            }
    
    def clear_cache(self):
        """Clear device cache/memory"""
        device_type = self.current_device.type
        
        if device_type == 'mps':
            if hasattr(torch.mps, 'empty_cache'):
                torch.mps.empty_cache()
                logger.info("Cleared MPS cache")
        elif device_type == 'cuda':
            torch.cuda.empty_cache()
            logger.info("Cleared CUDA cache")
    
    def to_device(self, tensor_or_model, **kwargs):
        """
        Move tensor or model to optimal device with error handling
        
        Args:
            tensor_or_model: PyTorch tensor or model
            **kwargs: Additional arguments for .to() method
            
        Returns:
            Tensor or model on target device
        """
        try:
            return tensor_or_model.to(self.current_device, **kwargs)
        except Exception as e:
            if self.fallback_enabled and self.current_device.type != 'cpu':
                logger.warning(f"Failed to move to {self.current_device.type}, falling back to CPU: {e}")
                return tensor_or_model.to('cpu', **kwargs)
            else:
                raise e
    
    def get_optimal_batch_size(self, model_memory_gb: float, safety_factor: float = 0.8) -> int:
        """
        Estimate optimal batch size based on available memory
        
        Args:
            model_memory_gb: Estimated model memory usage in GB
            safety_factor: Safety margin (0.8 = use 80% of available memory)
            
        Returns:
            Recommended batch size
        """
        memory_info = self.get_memory_info()
        available_memory = memory_info.get('available_gb', memory_info.get('free_gb', 4.0))
        
        # Reserve some memory for the system and other processes
        usable_memory = available_memory * safety_factor
        
        # Estimate batch size (very rough heuristic)
        if model_memory_gb > 0:
            estimated_batch_size = max(1, int(usable_memory / model_memory_gb))
        else:
            estimated_batch_size = 32  # Default fallback
            
        logger.info(f"Recommended batch size: {estimated_batch_size} "
                   f"(based on {usable_memory:.1f}GB usable memory)")
        
        return estimated_batch_size
    
    def print_device_summary(self):
        """Print a comprehensive summary of available devices"""
        print("\n" + "="*60)
        print("🖥️  MAC GPU ACCELERATION DEVICE SUMMARY")
        print("="*60)
        
        for device_type, info in self.device_info.items():
            status = "✅" if info.is_available else "❌"
            current = "👈 CURRENT" if info.device == self.current_device else ""
            
            print(f"\n{status} {device_type.upper()} {current}")
            print(f"   Memory: {info.memory_gb:.1f} GB")
            print(f"   Compute: {info.compute_capability}")
            print(f"   Performance Score: {info.performance_score:.1f}")
            
        print(f"\n🎯 Current Device: {self.current_device}")
        
        # Memory status
        memory_info = self.get_memory_info()
        print(f"💾 Memory Status: {memory_info.get('used_gb', 0):.1f}GB used / "
              f"{memory_info.get('total_gb', 0):.1f}GB total "
              f"({memory_info.get('percent_used', 0):.1f}%)")
        
        print("="*60)
    
    def benchmark_device(self, matrix_size: int = 2048, num_iterations: int = 5) -> Dict[str, float]:
        """
        Benchmark current device performance
        
        Args:
            matrix_size: Size of matrices for multiplication test
            num_iterations: Number of test iterations
            
        Returns:
            Performance metrics
        """
        import time
        
        logger.info(f"Benchmarking {self.current_device.type} device...")
        
        # Create test matrices
        x = torch.randn(matrix_size, matrix_size, device=self.current_device)
        y = torch.randn(matrix_size, matrix_size, device=self.current_device)
        
        # Warmup
        for _ in range(2):
            _ = torch.matmul(x, y)
            if self.current_device.type == 'mps':
                torch.mps.synchronize()
            elif self.current_device.type == 'cuda':
                torch.cuda.synchronize()
        
        # Benchmark
        times = []
        for i in range(num_iterations):
            start_time = time.time()
            result = torch.matmul(x, y)
            
            # Synchronize to ensure computation is complete
            if self.current_device.type == 'mps':
                torch.mps.synchronize()
            elif self.current_device.type == 'cuda':
                torch.cuda.synchronize()
                
            end_time = time.time()
            times.append(end_time - start_time)
            
        avg_time = sum(times) / len(times)
        min_time = min(times)
        max_time = max(times)
        
        # Calculate GFLOPS (rough estimate)
        ops = 2 * matrix_size**3  # Matrix multiplication operations
        gflops = (ops / avg_time) / 1e9
        
        metrics = {
            'avg_time_s': avg_time,
            'min_time_s': min_time,
            'max_time_s': max_time,
            'gflops': gflops,
            'matrix_size': matrix_size,
            'iterations': num_iterations
        }
        
        logger.info(f"Benchmark results: {avg_time:.4f}s avg, {gflops:.1f} GFLOPS")
        return metrics
    
    def _estimate_apple_silicon_performance(self) -> float:
        """Estimate Apple Silicon performance score"""
        chip = self._get_apple_silicon_chip()
        
        # Performance scores based on real-world benchmarks
        performance_map = {
            'M4': 8.5,   # Latest generation
            'M3': 7.0,   
            'M2': 5.0,   
            'M1': 3.5,   # Base score
        }
        
        base_score = performance_map.get(chip[:2], 3.0)
        
        # Adjust for Pro/Max variants
        if 'Pro' in chip:
            base_score *= 1.4
        elif 'Max' in chip:
            base_score *= 1.8
        elif 'Ultra' in chip:
            base_score *= 2.5
            
        return base_score
    
    def _get_apple_silicon_chip(self) -> str:
        """Get detailed Apple Silicon chip information"""
        try:
            import subprocess
            result = subprocess.run(['sysctl', '-n', 'machdep.cpu.brand_string'], 
                                 capture_output=True, text=True)
            brand_string = result.stdout.strip()
            
            # Extract chip model
            if 'M4' in brand_string:
                if 'Pro' in brand_string:
                    return 'M4 Pro'
                elif 'Max' in brand_string:
                    return 'M4 Max'
                else:
                    return 'M4'
            elif 'M3' in brand_string:
                if 'Pro' in brand_string:
                    return 'M3 Pro'
                elif 'Max' in brand_string:
                    return 'M3 Max'
                else:
                    return 'M3'
            elif 'M2' in brand_string:
                if 'Ultra' in brand_string:
                    return 'M2 Ultra'
                elif 'Pro' in brand_string:
                    return 'M2 Pro'
                elif 'Max' in brand_string:
                    return 'M2 Max'
                else:
                    return 'M2'
            elif 'M1' in brand_string:
                if 'Ultra' in brand_string:
                    return 'M1 Ultra'
                elif 'Pro' in brand_string:
                    return 'M1 Pro'
                elif 'Max' in brand_string:
                    return 'M1 Max'
                else:
                    return 'M1'
                    
        except Exception:
            pass
            
        return 'Apple Silicon'
    
    def is_low_memory(self, threshold_percent: float = 80.0) -> bool:
        """Check if device memory usage is above threshold"""
        memory_info = self.get_memory_info()
        return memory_info.get('percent_used', 0) > threshold_percent
    
    def get_recommended_settings(self, workload_type: str = 'general') -> Dict[str, Any]:
        """
        Get recommended settings for different workload types
        
        Args:
            workload_type: 'training', 'inference', 'general'
            
        Returns:
            Dictionary of recommended settings
        """
        device_type = self.current_device.type
        memory_gb = self.get_device_info().memory_gb
        
        settings = {
            'device': self.current_device,
            'mixed_precision': False,
            'batch_size': 32,
            'num_workers': 4,
            'pin_memory': False
        }
        
        if device_type == 'mps':
            settings.update({
                'mixed_precision': True,  # MPS benefits from mixed precision
                'pin_memory': False,      # Not needed with unified memory
                'num_workers': min(8, psutil.cpu_count()),
            })
            
            if workload_type == 'training':
                settings['batch_size'] = self.get_optimal_batch_size(1.0)  # Estimate 1GB per model
            elif workload_type == 'inference':
                settings['batch_size'] = min(64, int(memory_gb * 8))  # More aggressive for inference
                
        elif device_type == 'cuda':
            settings.update({
                'mixed_precision': True,
                'pin_memory': True,
                'num_workers': min(8, psutil.cpu_count()),
            })
            
        elif device_type == 'cpu':
            settings.update({
                'mixed_precision': False,  # CPU doesn't support mixed precision
                'num_workers': psutil.cpu_count(),
                'batch_size': min(16, int(memory_gb))  # Conservative for CPU
            })
        
        return settings


# Singleton instance for easy access
_device_manager = None

def get_device_manager(**kwargs) -> DeviceManager:
    """Get singleton device manager instance"""
    global _device_manager
    if _device_manager is None:
        _device_manager = DeviceManager(**kwargs)
    return _device_manager

def get_device() -> torch.device:
    """Quick access to optimal device"""
    return get_device_manager().get_device()

def to_device(tensor_or_model, **kwargs):
    """Quick access to move tensors/models to optimal device"""
    return get_device_manager().to_device(tensor_or_model, **kwargs)


# Context manager for temporary device switching
class temporary_device:
    """Context manager for temporary device switching"""
    
    def __init__(self, device_type: str):
        self.device_type = device_type
        self.manager = get_device_manager()
        self.original_device = None
        
    def __enter__(self):
        self.original_device = self.manager.current_device
        if not self.manager.switch_device(self.device_type):
            raise RuntimeError(f"Cannot switch to device: {self.device_type}")
        return self.manager.current_device
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.manager.current_device = self.original_device


if __name__ == "__main__":
    # Example usage and testing
    manager = DeviceManager()
    manager.print_device_summary()
    
    # Benchmark current device
    results = manager.benchmark_device()
    print(f"\nBenchmark Results: {results}")
    
    # Test memory monitoring
    memory_info = manager.get_memory_info()
    print(f"\nMemory Info: {memory_info}")
    
    # Get recommended settings
    settings = manager.get_recommended_settings('training')
    print(f"\nRecommended Training Settings: {settings}")