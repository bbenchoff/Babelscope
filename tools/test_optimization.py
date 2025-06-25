#!/usr/bin/env python3
"""
Quick GPU Optimization Test
Tests the optimized parameters on a small batch
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'emulators'))

def test_optimization():
    print("TESTING GPU OPTIMIZATION")
    print("=" * 40)
    
    try:
        from parallel_chip8_sorting import CUDASortingDetector
        print("Successfully imported optimized detector")
        
        # Test with small batch
        test_batch_size = 5000
        detector = CUDASortingDetector(test_batch_size)
        
        print(f"Test configuration:")
        print(f"  Batch size: {test_batch_size:,}")
        print(f"  Block size: {detector.block_size}")
        print(f"  Grid size: {detector.grid_size}")
        
        # Generate test ROMs
        import numpy as np
        test_roms = []
        for i in range(100):  # Small test
            rom = np.random.randint(0, 256, size=3584, dtype=np.uint8)
            test_roms.append(rom)
        
        print("Loading test ROMs...")
        detector.load_roms_with_sort_arrays(test_roms)
        
        print("Running short test (10,000 cycles)...")
        import time
        start_time = time.time()
        
        sorts_found = detector.run_sorting_detection(cycles=10000, sort_check_interval=1000)
        
        test_time = time.time() - start_time
        
        print(f"\nTest Results:")
        print(f"  Execution time: {test_time:.2f}s")
        print(f"  ROMs processed: {test_batch_size:,}")
        print(f"  Rate: {test_batch_size/test_time:.0f} ROMs/sec")
        print(f"  Sorts found: {sorts_found}")
        print("\nOptimization test PASSED!")
        
        return True
        
    except Exception as e:
        print(f"Test failed: {e}")
        return False

if __name__ == "__main__":
    success = test_optimization()
    if success:
        print("\nREADY TO RUN FULL SEARCH:")
        print("python launch_optimized.py")
    else:
        print("\nOptimization test failed - check your setup")
    
    exit(0 if success else 1)
