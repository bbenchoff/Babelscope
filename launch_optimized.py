#!/usr/bin/env python3
"""
Optimized Babelscope Launcher
Auto-configured for maximum GPU utilization on NVIDIA GeForce GTX 1070
"""

import subprocess
import sys

def main():
    print("OPTIMIZED BABELSCOPE SORTING SEARCH")
    print("=" * 50)
    print(f"GPU: NVIDIA GeForce GTX 1070")
    print(f"Optimized batch size: 39,999")
    print(f"Optimized block size: 512")
    print(f"Expected GPU utilization: 100.0%")
    print("=" * 50)
    
    # Launch with optimal parameters
    cmd = [
        sys.executable, "run_sorting_search.py",
        "--batch-size", str(39999),
        "--continuous",
        "--cycles", "50000"
    ]
    
    print(f"Launching: {' '.join(cmd)}")
    
    try:
        subprocess.run(cmd, check=True)
    except KeyboardInterrupt:
        print("\nSearch interrupted by user")
    except subprocess.CalledProcessError as e:
        print(f"Launch failed: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
