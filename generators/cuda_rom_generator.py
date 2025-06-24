"""
CUDA-Optimized Random ROM Generator for Babelscope
Generates completely random CHIP-8 ROMs using GPU acceleration
Simple and fast - no special logic, just random bytes
"""

import cupy as cp
import numpy as np
from typing import List, Optional
import time

class CUDAROMGenerator:
    """
    Simple CUDA-accelerated random ROM generator
    Generates thousands of random CHIP-8 ROMs simultaneously on GPU
    """
    
    def __init__(self, rom_size: int = 3584, seed: Optional[int] = None):
        self.rom_size = rom_size
        
        if seed is not None:
            cp.random.seed(seed)
            np.random.seed(seed)
        
        # Get GPU info
        current_device = cp.cuda.Device()
        props = cp.cuda.runtime.getDeviceProperties(current_device.id)
        gpu_name = props['name'].decode()
        gpu_memory_gb = props['totalGlobalMem'] / 1024**3
        
        print(f"CUDA ROM Generator on {gpu_name} ({gpu_memory_gb:.1f} GB)")
        print(f"ROM size: {rom_size} bytes")
    
    def generate_batch(self, count: int) -> List[np.ndarray]:
        """Generate a batch of completely random ROMs using CUDA"""
        print(f"Generating {count:,} random ROMs on GPU...")
        start_time = time.time()
        
        # Generate all ROMs at once on GPU - simple and fast
        all_roms_gpu = cp.random.randint(
            0, 256, 
            size=(count, self.rom_size), 
            dtype=cp.uint8
        )
        
        # Convert to CPU and create list
        all_roms_cpu = cp.asnumpy(all_roms_gpu)
        rom_list = [all_roms_cpu[i] for i in range(count)]
        
        generation_time = time.time() - start_time
        roms_per_second = count / generation_time if generation_time > 0 else 0
        
        print(f"✅ Generated {count:,} ROMs in {generation_time:.2f}s ({roms_per_second:.0f} ROMs/sec)")
        
        return rom_list
    
    def generate_single(self) -> np.ndarray:
        """Generate a single random ROM"""
        return cp.asnumpy(cp.random.randint(0, 256, size=self.rom_size, dtype=cp.uint8))


def generate_random_roms_cuda(count: int, rom_size: int = 3584) -> List[np.ndarray]:
    """
    Convenience function for generating random ROMs with CUDA
    Compatible with existing sorting detector interface
    """
    generator = CUDAROMGenerator(rom_size)
    return generator.generate_batch(count)


def generate_random_roms(count: int, rom_size: int = 3584) -> List[np.ndarray]:
    """
    CPU fallback for compatibility
    """
    print(f"🎲 Generating {count:,} random ROMs on CPU...")
    start_time = time.time()
    
    roms = []
    for i in range(count):
        rom_data = np.random.randint(0, 256, size=rom_size, dtype=np.uint8)
        roms.append(rom_data)
    
    generation_time = time.time() - start_time
    roms_per_second = count / generation_time if generation_time > 0 else 0
    
    print(f"✅ Generated {count:,} ROMs in {generation_time:.2f}s ({roms_per_second:.0f} ROMs/sec)")
    
    return roms


if __name__ == "__main__":
    # Test the generator
    generator = CUDAROMGenerator()
    
    # Test small batch
    roms = generator.generate_batch(1000)
    print(f"Generated {len(roms)} ROMs, each {len(roms[0])} bytes")
    
    # Test large batch
    start = time.time()
    large_batch = generator.generate_batch(50000)
    end = time.time()
    print(f"Large batch: {len(large_batch):,} ROMs in {end-start:.2f}s")