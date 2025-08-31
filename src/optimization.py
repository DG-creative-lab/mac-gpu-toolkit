 """
Performance Optimization Module for Mac GPU Acceleration Toolkit
Advanced optimization techniques for Apple Silicon and cross-platform ML workloads
"""

import torch
import torch.nn as nn
import time
import gc
import logging
from typing import Optional, Dict, Any, Callable, Tuple, Union
from contextlib import contextmanager
from functools import wraps
import threading
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class OptimizationProfile:
    """Performance optimization profile"""
    mixed_precision: bool = True
    gradient_checkpointing: bool = False
    compile_model: bool = False
    memory_efficient: bool = True
    batch_size_scaling: float = 1.0
    num_workers: int = 4


class PerformanceOptimizer:
    """
    Advanced performance optimization for Mac GPU acceleration
    """
    
    def __init__(self, device_manager=None):
        """Initialize optimizer with device manager"""
        if device_manager is None:
            from .device_manager import get_device_manager
            self.device_manager = get_device_manager()
        else:
            self.device_manager = device_manager
            
        self.device = self.device_manager.get_device()
        self.device_type = self.device.type
        
    def get_optimization_profile(self, workload_type: str = 'general') -> OptimizationProfile:
        """
        Get optimization profile based on device and workload type
        
        Args:
            workload_type: 'training', 'inference', 'batch_processing'
        """
        profile = OptimizationProfile()
        
        if self.device_type == 'mps':
            # Apple Silicon optimizations
            profile.mixed_precision = True  # MPS benefits significantly from mixed precision
            profile.memory_efficient = True
            profile.compile_model = False  # torch.compile not yet stable on MPS
            
            if workload_type == 'training':
                profile.gradient_checkpointing = True  # Save memory during training
                profile.batch_size_scaling = 1.2  # Can handle slightly larger batches
            elif workload_type == 'inference':
                profile.batch_size_scaling = 1.5  # More aggressive for inference
            elif workload_type == 'batch_processing':
                profile.batch_size_scaling = 0.8  # Conservative for large batches
                profile.num_workers = 8
                
        elif self.device_type == 'cuda':
            # NVIDIA GPU optimizations
            profile.mixed_precision = True
            profile.compile_model = True  # torch.compile works well on CUDA
            
            if workload_type == 'training':
                profile.gradient_checkpointing = True
                profile.batch_size_scaling = 1.0
            elif workload_type == 'inference':
                profile.batch_size_scaling = 2.0  # CUDA can handle larger inference batches
                
        else:  # CPU
            profile.mixed_precision = False  # CPU doesn't support mixed precision
            profile.memory_efficient = False
            profile.batch_size_scaling = 0.5  # Smaller batches for CPU
            profile.num_workers = min(8, torch.get_num_threads())
            
        return profile
    
    def optimize_model(self, model: nn.Module, profile: Optional[OptimizationProfile] = None) -> nn.Module:
        """
        Apply comprehensive model optimizations
        
        Args:
            model: PyTorch model to optimize
            profile: Optimization profile to use
            
        Returns:
            Optimized model
        """
        if profile is None:
            profile = self.get_optimization_profile()
            
        logger.info(f"Optimizing model for {self.device_type} device...")
        
        # Move to device
        model = model.to(self.device)
        
        # Apply mixed precision
        if profile.mixed_precision and self.device_type in ['mps', 'cuda']:
            model = model.half()
            logger.info("Applied mixed precision (float16)")
            
        # Apply gradient checkpointing for training
        if profile.gradient_checkpointing and hasattr(model, 'gradient_checkpointing_enable'):
            model.gradient_checkpointing_enable()
            logger.info("Enabled gradient checkpointing")
            
        # Apply model compilation (if supported)
        if profile.compile_model and hasattr(torch, 'compile'):
            try:
                model = torch.compile(model, mode='default')
                logger.info("Applied torch.compile optimization")
            except Exception as e:
                logger.warning(f"torch.compile failed: {e}")
                
        # Memory-efficient attention (if available)
        if profile.memory_efficient:
            self._apply_memory_efficient_attention(model)
            
        return model
    
    def _apply_memory_efficient_attention(self, model: nn.Module):
        """Apply memory-efficient attention where possible"""
        try:
            # For transformers with flash attention support
            if hasattr(model, 'config') and hasattr(model.config, 'use_flash_attention_2'):
                model.config.use_flash_attention_2 = True
                logger.info("Enabled flash attention 2")
        except Exception as e:
            logger.debug(f"Could not enable memory-efficient attention: {e}")
    
    def optimize_dataloader(self, batch_size: int, num_workers: Optional[int] = None, 
                          profile: Optional[OptimizationProfile] = None) -> Dict[str, Any]:
        """
        Get optimized DataLoader parameters
        
        Args:
            batch_size: Base batch size
            num_workers: Number of worker processes
            profile: Optimization profile
            
        Returns:
            Optimized DataLoader parameters
        """
        if profile is None:
            profile = self.get_optimization_profile()
            
        if num_workers is None:
            num_workers = profile.num_workers
            
        # Scale batch size based on device and profile
        optimized_batch_size = int(batch_size * profile.batch_size_scaling)
        
        # Device-specific optimizations
        params = {
            'batch_size': optimized_batch_size,
            'num_workers': num_workers,
            'pin_memory': self.device_type == 'cuda',  # Only beneficial for CUDA
            'persistent_workers': num_workers > 0,
        }
        
        # Apple Silicon specific optimizations
        if self.device_type == 'mps':
            # Unified memory architecture doesn't need pin_memory
            params['pin_memory'] = False
            # Prefetch can help with MPS
            params['prefetch_factor'] = 2 if num_workers > 0 else None
            
        logger.info(f"Optimized DataLoader: batch_size={optimized_batch_size}, "
                   f"num_workers={num_workers}, pin_memory={params['pin_memory']}")
        
        return params


@contextmanager
def memory_efficient_inference():
    """
    Context manager for memory-efficient inference
    Automatically manages memory and caching
    """
    original_grad_state = torch.is_grad_enabled()
    torch.set_grad_enabled(False)
    
    try:
        yield
    finally:
        torch.set_grad_enabled(original_grad_state)
        # Clear cache based on device
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
        elif torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()


class AdaptiveBatchProcessor:
    """
    Adaptive batch processing that automatically adjusts batch size based on memory
    """
    
    def __init__(self, device_manager=None, initial_batch_size: int = 32, 
                 memory_threshold: float = 0.85):
        """
        Initialize adaptive batch processor
        
        Args:
            device_manager: Device manager instance
            initial_batch_size: Starting batch size
            memory_threshold: Memory usage threshold for adjustment (0.85 = 85%)
        """
        if device_manager is None:
            from .device_manager import get_device_manager
            self.device_manager = get_device_manager()
        else:
            self.device_manager = device_manager
            
        self.current_batch_size = initial_batch_size
        self.memory_threshold = memory_threshold
        self.min_batch_size = 1
        self.max_batch_size = initial_batch_size * 4
        self.adjustment_history = []
        
    def process_batch(self, process_fn: Callable, data_batch: Any, 
                     auto_adjust: bool = True) -> Any:
        """
        Process a batch with automatic memory management
        
        Args:
            process_fn: Function to process the batch
            data_batch: Batch data
            auto_adjust: Whether to automatically adjust batch size
            
        Returns:
            Processing results
        """
        try:
            # Check memory before processing
            if auto_adjust:
                self._check_and_adjust_memory()
                
            # Process the batch
            with memory_efficient_inference():
                result = process_fn(data_batch)
                
            # Monitor memory after processing
            if auto_adjust:
                self._monitor_post_processing()
                
            return result
            
        except torch.cuda.OutOfMemoryError:
            if auto_adjust and self.current_batch_size > self.min_batch_size:
                self._reduce_batch_size()
                logger.warning(f"CUDA OOM: Reduced batch size to {self.current_batch_size}")
                raise
            else:
                raise
        except RuntimeError as e:
            if "MPS" in str(e) and "memory" in str(e).lower():
                if auto_adjust and self.current_batch_size > self.min_batch_size:
                    self._reduce_batch_size()
                    logger.warning(f"MPS Memory Error: Reduced batch size to {self.current_batch_size}")
                    raise
            else:
                raise
    
    def _check_and_adjust_memory(self):
        """Check memory usage and adjust batch size if needed"""
        memory_info = self.device_manager.get_memory_info()
        memory_percent = memory_info.get('percent_used', 0)
        
        if memory_percent > self.memory_threshold * 100:
            self._reduce_batch_size()
        elif memory_percent < (self.memory_threshold * 0.7) * 100:
            self._increase_batch_size()
    
    def _monitor_post_processing(self):
        """Monitor memory after processing and learn for next batch"""
        memory_info = self.device_manager.get_memory_info()
        memory_percent = memory_info.get('percent_used', 0)
        
        self.adjustment_history.append({
            'batch_size': self.current_batch_size,
            'memory_percent': memory_percent,
            'timestamp': time.time()
        })
        
        # Keep only recent history
        if len(self.adjustment_history) > 10:
            self.adjustment_history = self.adjustment_history[-10:]
    
    def _reduce_batch_size(self):
        """Reduce batch size with minimum constraints"""
        self.current_batch_size = max(self.min_batch_size, 
                                    int(self.current_batch_size * 0.75))
        self.device_manager.clear_cache()
        
    def _increase_batch_size(self):
        """Increase batch size with maximum constraints"""
        self.current_batch_size = min(self.max_batch_size,
                                    int(self.current_batch_size * 1.25))
    
    def get_optimal_batch_size(self) -> int:
        """Get current optimal batch size"""
        return self.current_batch_size


class PerformanceProfiler:
    """
    Performance profiling and monitoring for Mac GPU workloads
    """
    
    def __init__(self):
        self.profiles = {}
        self.current_profile = None
        
    @contextmanager
    def profile(self, operation_name: str):
        """
        Profile a specific operation
        
        Args:
            operation_name: Name of the operation being profiled
        """
        start_time = time.time()
        start_memory = self._get_memory_usage()
        
        try:
            yield
        finally:
            end_time = time.time()
            end_memory = self._get_memory_usage()
            
            profile_data = {
                'duration': end_time - start_time,
                'memory_start': start_memory,
                'memory_end': end_memory,
                'memory_delta': end_memory - start_memory,
                'timestamp': start_time
            }
            
            if operation_name not in self.profiles:
                self.profiles[operation_name] = []
            self.profiles[operation_name].append(profile_data)
            
            logger.debug(f"Profile {operation_name}: {profile_data['duration']:.4f}s, "
                        f"Memory: {profile_data['memory_delta']:.2f}MB")
    
    def _get_memory_usage(self) -> float:
        """Get current memory usage in MB"""
        if torch.cuda.is_available():
            return torch.cuda.memory_allocated() / 1024**2
        elif torch.backends.mps.is_available():
            # For MPS, approximate using system memory
            import psutil
            return psutil.virtual_memory().used / 1024**2
        else:
            import psutil
            return psutil.virtual_memory().used / 1024**2
    
    def get_profile_summary(self, operation_name: str) -> Dict[str, float]:
        """Get summary statistics for a profiled operation"""
        if operation_name not in self.profiles:
            return {}
            
        data = self.profiles[operation_name]
        durations = [p['duration'] for p in data]
        memory_deltas = [p['memory_delta'] for p in data]
        
        return {
            'count': len(data),
            'avg_duration': sum(durations) / len(durations),
            'min_duration': min(durations),
            'max_duration': max(durations),
            'avg_memory_delta': sum(memory_deltas) / len(memory_deltas),
            'total_time': sum(durations)
        }
    
    def print_profile_report(self):
        """Print a comprehensive performance report"""
        print("\n" + "="*60)
        print("📊 PERFORMANCE PROFILE REPORT")
        print("="*60)
        
        for operation_name in sorted(self.profiles.keys()):
            summary = self.get_profile_summary(operation_name)
            print(f"\n🔍 {operation_name}")
            print(f"   Executions: {summary['count']}")
            print(f"   Avg Time: {summary['avg_duration']:.4f}s")
            print(f"   Total Time: {summary['total_time']:.2f}s")
            print(f"   Avg Memory: {summary['avg_memory_delta']:.2f}MB")
            
        print("="*60)


class MacSpecificOptimizer:
    """
    Mac-specific optimization techniques
    """
    
    @staticmethod
    def setup_mps_environment():
        """Setup optimal environment variables for MPS"""
        import os
        
        # MPS specific optimizations
        env_vars = {
            'PYTORCH_ENABLE_MPS_FALLBACK': '1',  # Enable fallback for unsupported ops
            'PYTORCH_MPS_HIGH_WATERMARK_RATIO': '0.0',  # Disable memory caching threshold
        }
        
        for key, value in env_vars.items():
            os.environ[key] = value
            
        logger.info("Applied MPS environment optimizations")
    
    @staticmethod
    def optimize_for_transformer_inference(model: nn.Module) -> nn.Module:
        """
        Apply transformer-specific optimizations for Apple Silicon
        
        Args:
            model: Transformer model to optimize
            
        Returns:
            Optimized model
        """
        # Set to inference mode
        model.eval()
        
        # Apply common optimizations
        for module in model.modules():
            # Optimize attention mechanisms
            if hasattr(module, 'flash_attention'):
                module.flash_attention = True
                
            # Optimize layer norm
            if isinstance(module, nn.LayerNorm):
                module.eps = 1e-5  # Optimal epsilon for Apple Silicon
                
        # Enable Apple Silicon specific optimizations
        if hasattr(model, 'config'):
            if hasattr(model.config, 'use_cache'):
                model.config.use_cache = True  # Enable KV caching
                
        logger.info("Applied transformer-specific optimizations")
        return model
    
    @staticmethod
    def setup_threading_for_mac():
        """Setup optimal threading configuration for Apple Silicon"""
        import os
        
        # Get optimal thread count
        cpu_count = torch.get_num_threads()
        
        # Apple Silicon specific threading
        if platform.system() == 'Darwin':
            # Performance cores vs efficiency cores optimization
            perf_cores = min(8, cpu_count)  # Assume max 8 performance cores
            
            # Set threading for various libraries
            os.environ['OMP_NUM_THREADS'] = str(perf_cores)
            os.environ['MKL_NUM_THREADS'] = str(perf_cores)
            os.environ['VECLIB_MAXIMUM_THREADS'] = str(perf_cores)
            
            torch.set_num_threads(perf_cores)
            
            logger.info(f"Optimized threading for Apple Silicon: {perf_cores} threads")


def performance_monitor(func: Callable) -> Callable:
    """
    Decorator for automatic performance monitoring
    
    Usage:
        @performance_monitor
        def my_training_step(batch):
            # Your training code here
            pass
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        profiler = PerformanceProfiler()
        
        with profiler.profile(func.__name__):
            result = func(*args, **kwargs)
            
        return result
    return wrapper


class BatchSizeOptimizer:
    """
    Automatically find optimal batch size for your hardware
    """
    
    def __init__(self, device_manager=None):
        if device_manager is None:
            from .device_manager import get_device_manager
            self.device_manager = get_device_manager()
        else:
            self.device_manager = device_manager
    
    def find_optimal_batch_size(self, model: nn.Module, sample_input: torch.Tensor,
                              target_memory_usage: float = 0.8) -> int:
        """
        Find optimal batch size through binary search
        
        Args:
            model: Model to test
            sample_input: Sample input tensor (batch_size=1)
            target_memory_usage: Target memory usage (0.8 = 80%)
            
        Returns:
            Optimal batch size
        """
        device = self.device_manager.get_device()
        model = model.to(device)
        sample_input = sample_input.to(device)
        
        # Start with binary search
        min_batch = 1
        max_batch = 256  # Conservative upper bound
        optimal_batch = 1
        
        logger.info("Finding optimal batch size...")
        
        with torch.no_grad():
            while min_batch <= max_batch:
                test_batch = (min_batch + max_batch) // 2
                
                try:
                    # Create test batch
                    batch_input = sample_input.repeat(test_batch, *([1] * (sample_input.dim() - 1)))
                    
                    # Test forward pass
                    _ = model(batch_input)
                    
                    # Check memory usage
                    memory_info = self.device_manager.get_memory_info()
                    memory_usage = memory_info.get('percent_used', 0) / 100
                    
                    if memory_usage < target_memory_usage:
                        optimal_batch = test_batch
                        min_batch = test_batch + 1
                    else:
                        max_batch = test_batch - 1
                        
                    # Clear memory
                    del batch_input
                    self.device_manager.clear_cache()
                    
                except (RuntimeError, torch.cuda.OutOfMemoryError) as e:
                    # OOM or other memory error
                    max_batch = test_batch - 1
                    self.device_manager.clear_cache()
                    
        logger.info(f"Optimal batch size found: {optimal_batch}")
        return optimal_batch


class ThroughputOptimizer:
    """
    Optimize throughput for different types of workloads
    """
    
    def __init__(self, device_manager=None):
        if device_manager is None:
            from .device_manager import get_device_manager
            self.device_manager = get_device_manager()
        else:
            self.device_manager = device_manager
    
    def optimize_for_llm_inference(self, model: nn.Module, tokenizer=None) -> Dict[str, Any]:
        """
        Optimize model for LLM inference throughput
        
        Args:
            model: Language model
            tokenizer: Optional tokenizer
            
        Returns:
            Optimization settings and modified model
        """
        device = self.device_manager.get_device()
        
        # Move model to device
        model = model.to(device)
        model.eval()
        
        optimizations = {
            'device': device,
            'torch_dtype': torch.float16 if device.type in ['mps', 'cuda'] else torch.float32,
            'low_cpu_mem_usage': True,
        }
        
        # Apply mixed precision
        if device.type in ['mps', 'cuda']:
            model = model.half()
            optimizations['use_mixed_precision'] = True
            
        # Enable KV caching if supported
        if hasattr(model, 'config'):
            if hasattr(model.config, 'use_cache'):
                model.config.use_cache = True
                optimizations['kv_cache_enabled'] = True
                
        # Apple Silicon specific optimizations
        if device.type == 'mps':
            MacSpecificOptimizer.setup_mps_environment()
            optimizations['mps_optimized'] = True
            
        logger.info("Applied LLM inference optimizations")
        return {'model': model, 'settings': optimizations}
    
    def create_inference_pipeline(self, model: nn.Module, preprocess_fn: Callable,
                                postprocess_fn: Callable) -> Callable:
        """
        Create an optimized inference pipeline
        
        Args:
            model: Model for inference
            preprocess_fn: Preprocessing function
            postprocess_fn: Postprocessing function
            
        Returns:
            Optimized pipeline function
        """
        device = self.device_manager.get_device()
        model = model.to(device)
        model.eval()
        
        def optimized_pipeline(inputs):
            with memory_efficient_inference():
                # Preprocess
                processed_inputs = preprocess_fn(inputs)
                
                # Move to device
                if isinstance(processed_inputs, torch.Tensor):
                    processed_inputs = processed_inputs.to(device, non_blocking=True)
                elif isinstance(processed_inputs, (list, tuple)):
                    processed_inputs = [
                        inp.to(device, non_blocking=True) if isinstance(inp, torch.Tensor) else inp
                        for inp in processed_inputs
                    ]
                
                # Inference
                with torch.no_grad():
                    outputs = model(processed_inputs)
                
                # Postprocess
                results = postprocess_fn(outputs)
                
                return results
                
        return optimized_pipeline


class MacGPUProfiler:
    """
    Specialized profiler for Mac GPU workloads
    """
    
    def __init__(self):
        self.enabled = False
        self.profile_data = {}
        
    def start_profiling(self):
        """Start performance profiling"""
        self.enabled = True
        self.profile_data = {}
        logger.info("Started Mac GPU profiling")
        
    def stop_profiling(self):
        """Stop performance profiling"""
        self.enabled = False
        logger.info("Stopped Mac GPU profiling")
        
    @contextmanager
    def profile_operation(self, name: str):
        """Profile a specific operation"""
        if not self.enabled:
            yield
            return
            
        # Record start state
        start_time = time.time()
        start_memory = self._get_current_memory()
        
        try:
            yield
        finally:
            # Record end state
            end_time = time.time()
            end_memory = self._get_current_memory()
            
            # Store profile data
            if name not in self.profile_data:
                self.profile_data[name] = []
                
            self.profile_data[name].append({
                'duration': end_time - start_time,
                'memory_used': end_memory - start_memory,
                'timestamp': start_time
            })
    
    def _get_current_memory(self) -> float:
        """Get current memory usage in MB"""
        if torch.backends.mps.is_available():
            # For MPS, we'll use system memory as proxy
            import psutil
            return psutil.virtual_memory().used / 1024**2
        elif torch.cuda.is_available():
            return torch.cuda.memory_allocated() / 1024**2
        else:
            import psutil
            return psutil.virtual_memory().used / 1024**2
    
    def export_profile_data(self, filename: str):
        """Export profile data to JSON file"""
        import json
        
        with open(filename, 'w') as f:
            json.dump(self.profile_data, f, indent=2)
            
        logger.info(f"Profile data exported to {filename}")


# Utility functions for common optimization tasks

def optimize_model_for_mac(model: nn.Module, workload_type: str = 'inference') -> nn.Module:
    """
    One-stop function to optimize any model for Mac
    
    Args:
        model: PyTorch model to optimize
        workload_type: 'training' or 'inference'
        
    Returns:
        Optimized model
    """
    optimizer = PerformanceOptimizer()
    profile = optimizer.get_optimization_profile(workload_type)
    
    # Apply Mac-specific optimizations
    MacSpecificOptimizer.setup_mps_environment()
    MacSpecificOptimizer.setup_threading_for_mac()
    
    # Optimize model
    optimized_model = optimizer.optimize_model(model, profile)
    
    return optimized_model


def create_optimized_dataloader(dataset, batch_size: int = 32, **kwargs):
    """
    Create an optimized DataLoader for Mac
    
    Args:
        dataset: PyTorch dataset
        batch_size: Base batch size
        **kwargs: Additional DataLoader arguments
        
    Returns:
        Optimized DataLoader
    """
    from torch.utils.data import DataLoader
    
    optimizer = PerformanceOptimizer()
    profile = optimizer.get_optimization_profile()
    
    # Get optimized parameters
    params = optimizer.optimize_dataloader(batch_size, profile=profile)
    
    # Merge with user-provided kwargs
    params.update(kwargs)
    
    return DataLoader(dataset, **params)


def benchmark_gpu_performance(matrix_size: int = 2048, iterations: int = 10) -> Dict[str, Any]:
    """
    Comprehensive GPU performance benchmark
    
    Args:
        matrix_size: Size of test matrices
        iterations: Number of test iterations
        
    Returns:
        Benchmark results
    """
    from .device_manager import get_device_manager
    
    manager = get_device_manager()
    results = {}
    
    # Test each available device
    for device_type, device_info in manager.device_info.items():
        if not device_info.is_available:
            continue
            
        logger.info(f"Benchmarking {device_type}...")
        
        # Switch to device
        original_device = manager.current_device
        manager.switch_device(device_type)
        
        # Run benchmark
        device_results = manager.benchmark_device(matrix_size, iterations)
        results[device_type] = device_results
        
        # Restore original device
        manager.current_device = original_device
    
    # Calculate relative performance
    if 'cpu' in results:
        cpu_time = results['cpu']['avg_time_s']
        for device_type in results:
            if device_type != 'cpu':
                speedup = cpu_time / results[device_type]['avg_time_s']
                results[device_type]['speedup_vs_cpu'] = speedup
                
    return results


# Context managers for optimization

@contextmanager
def optimized_inference(model: nn.Module, profile: Optional[OptimizationProfile] = None):
    """
    Context manager for optimized inference
    
    Args:
        model: Model to optimize
        profile: Optimization profile
    """
    optimizer = PerformanceOptimizer()
    
    if profile is None:
        profile = optimizer.get_optimization_profile('inference')
    
    # Store original state
    original_training = model.training
    original_grad = torch.is_grad_enabled()
    
    try:
        # Set inference mode
        model.eval()
        torch.set_grad_enabled(False)
        
        # Apply optimizations
        with memory_efficient_inference():
            yield model
            
    finally:
        # Restore original state
        model.train(original_training)
        torch.set_grad_enabled(original_grad)


@contextmanager
def mac_performance_mode():
    """
    Context manager that applies all Mac-specific optimizations
    """
    # Setup optimal environment
    MacSpecificOptimizer.setup_mps_environment()
    MacSpecificOptimizer.setup_threading_for_mac()
    
    # Start profiling
    profiler = MacGPUProfiler()
    profiler.start_profiling()
    
    try:
        yield profiler
    finally:
        profiler.stop_profiling()


# Memory management utilities

def aggressive_memory_cleanup():
    """Aggressive memory cleanup for Mac GPU"""
    # Python garbage collection
    gc.collect()
    
    # PyTorch cache cleanup
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        
    logger.info("Performed aggressive memory cleanup")


def monitor_memory_usage(threshold: float = 0.9, auto_cleanup: bool = True) -> Callable:
    """
    Decorator to monitor memory usage during function execution
    
    Args:
        threshold: Memory threshold for warnings (0.9 = 90%)
        auto_cleanup: Whether to automatically cleanup on high usage
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            from .device_manager import get_device_manager
            
            manager = get_device_manager()
            
            # Check memory before
            memory_before = manager.get_memory_info()
            
            try:
                result = func(*args, **kwargs)
                
                # Check memory after
                memory_after = manager.get_memory_info()
                usage_percent = memory_after.get('percent_used', 0) / 100
                
                if usage_percent > threshold:
                    logger.warning(f"High memory usage: {usage_percent:.1%}")
                    
                    if auto_cleanup:
                        aggressive_memory_cleanup()
                        logger.info("Performed automatic memory cleanup")
                
                return result
                
            except (RuntimeError, torch.cuda.OutOfMemoryError) as e:
                if "memory" in str(e).lower():
                    logger.error(f"Memory error in {func.__name__}: {e}")
                    aggressive_memory_cleanup()
                raise
                
        return wrapper
    return decorator


if __name__ == "__main__":
    # Example usage and testing
    print("Testing Mac GPU Optimization Module...")
    
    # Test device manager integration
    from .device_manager import get_device_manager
    manager = get_device_manager()
    
    # Create optimizer
    optimizer = PerformanceOptimizer(manager)
    
    # Test optimization profile
    profile = optimizer.get_optimization_profile('training')
    print(f"Training profile: {profile}")
    
    # Test profiler
    profiler = PerformanceProfiler()
    profiler.start_profiling()
    
    with profiler.profile("test_operation"):
        time.sleep(0.1)  # Simulate work
        
    profiler.stop_profiling()
    profiler.print_profile_report()
    
    # Test benchmark
    results = benchmark_gpu_performance(1024, 3)
    print(f"Benchmark results: {results}")
    
    print("✅ Optimization module test complete!")