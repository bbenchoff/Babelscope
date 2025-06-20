"""
Pure Random CHIP-8 ROM Generator
Generates completely random data - no heuristics, no filtering, pure randomness
True computational archaeology approach
"""

import cupy as cp
import numpy as np
from typing import List
import time
import os

# CHIP-8 Constants
PROGRAM_START = 0x200
MEMORY_SIZE = 4096
MAX_PROGRAM_SIZE = MEMORY_SIZE - PROGRAM_START  # 3584 bytes available for programs
DEFAULT_ROM_SIZE = MAX_PROGRAM_SIZE  # Use full available space by default

# GPU kernel for generating purely random data
RANDOM_DATA_KERNEL = r'''
extern "C" __global__
void generate_random_data(
    unsigned char* rom_data,        // [instances][rom_size]
    unsigned int* rng_states,       // [instances] - RNG state per instance
    int num_instances,
    int rom_size,
    int seed_offset
) {
    int instance = blockIdx.x * blockDim.x + threadIdx.x;
    if (instance >= num_instances) return;
    
    // Initialize RNG state for this instance
    unsigned int state = rng_states[instance] + seed_offset;
    
    int base_idx = instance * rom_size;
    
    // Generate completely random bytes - no filtering, no validation
    for (int i = 0; i < rom_size; i++) {
        state = state * 1664525 + 1013904223;  // LCG
        rom_data[base_idx + i] = (unsigned char)(state >> 16);
    }
    
    // Update RNG state
    rng_states[instance] = state;
}
'''


class PureRandomChip8Generator:
    """
    Pure random CHIP-8 ROM generator - no heuristics, no filtering
    Generates completely random bytes and lets the emulator decide what happens
    """
    
    def __init__(self, rom_size: int = DEFAULT_ROM_SIZE):
        """
        Initialize the pure random ROM generator
        
        Args:
            rom_size: Size of each ROM in bytes
        """
        self.rom_size = rom_size
        
        # Compile the random data generation kernel
        self.random_kernel = cp.RawKernel(RANDOM_DATA_KERNEL, 'generate_random_data')
        
        print(f"Pure Random CHIP-8 Generator: ROM size = {rom_size} bytes")
        print("No heuristics - pure computational archaeology")
    
    def generate_batch(self, num_roms: int, seed_offset: int = 0) -> cp.ndarray:
        """
        Generate a batch of completely random ROM data
        
        Args:
            num_roms: Number of ROMs to generate
            seed_offset: Offset for RNG seeding
            
        Returns:
            CuPy array of shape (num_roms, rom_size) with pure random data
        """
        # Calculate grid/block dimensions
        block_size = min(256, num_roms)
        grid_size = (num_roms + block_size - 1) // block_size
        
        # Allocate memory
        rom_data = cp.zeros((num_roms, self.rom_size), dtype=cp.uint8)
        rng_states = cp.random.randint(1, 2**32, size=num_roms, dtype=cp.uint32)
        
        # Launch kernel to generate pure random data
        self.random_kernel(
            (grid_size,), (block_size,),
            (
                rom_data,
                rng_states,
                num_roms,
                self.rom_size,
                seed_offset
            )
        )
        
        cp.cuda.Stream.null.synchronize()
        return rom_data
    
    def generate_and_save(self, num_roms: int, output_dir: str = "random_roms", 
                         prefix: str = "random") -> List[str]:
        """
        Generate random ROMs and save them to disk
        
        Args:
            num_roms: Number of ROMs to generate
            output_dir: Directory to save ROMs
            prefix: Filename prefix
            
        Returns:
            List of saved filenames
        """
        print(f"Generating {num_roms:,} pure random ROMs...")
        start_time = time.time()
        
        # Generate random data
        roms = self.generate_batch(num_roms)
        
        generation_time = time.time() - start_time
        rate = num_roms / generation_time if generation_time > 0 else 0
        
        print(f"Generated {num_roms:,} ROMs in {generation_time:.2f}s ({rate:,.0f} ROMs/sec)")
        
        # Save to disk
        return self.save_roms(roms, output_dir, prefix)
    
    def save_roms(self, roms: cp.ndarray, output_dir: str = "random_roms", 
                  prefix: str = "random") -> List[str]:
        """
        Save ROMs to disk
        
        Args:
            roms: ROM data to save
            output_dir: Directory to save ROMs
            prefix: Filename prefix
            
        Returns:
            List of saved filenames
        """
        os.makedirs(output_dir, exist_ok=True)
        
        roms_cpu = cp.asnumpy(roms)
        saved_files = []
        
        for i, rom in enumerate(roms_cpu):
            filename = f"{prefix}_{i:06d}.ch8"
            filepath = os.path.join(output_dir, filename)
            
            with open(filepath, 'wb') as f:
                f.write(rom.tobytes())
            
            saved_files.append(filename)
        
        print(f"Saved {len(saved_files)} ROMs to {output_dir}/")
        return saved_files


def main():
    """Generate a batch of pure random CHIP-8 ROMs"""
    print("Pure Random CHIP-8 ROM Generator")
    print("Computational Archaeology - No Heuristics")
    print("=" * 50)
    
    # Create generator using full available memory space
    generator = PureRandomChip8Generator(rom_size=MAX_PROGRAM_SIZE)
    
    # Generate a batch of random ROMs
    num_roms = 10000  # Start with 10K pure random ROMs
    
    try:
        saved_files = generator.generate_and_save(num_roms, "output/random_roms")
        
        print(f"\nGenerated {len(saved_files)} pure random ROMs")
        print(f"Each ROM is {MAX_PROGRAM_SIZE} bytes of completely random data")
        
    except KeyboardInterrupt:
        print("\nGeneration stopped by user")


if __name__ == "__main__":
    main()