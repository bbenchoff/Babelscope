#!/usr/bin/env python3
"""
Test runner for the parallel CHIP-8 emulator
Loads and runs CHIP-8 ROMs on GPU, saves display outputs as PNGs
"""

import os
import sys
import glob
import argparse
import time
from pathlib import Path

# Add the emulators directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'emulators'))

try:
    from mega_kernel_chip8 import MegaKernelChip8Emulator as ParallelChip8Emulator
    print("Using mega-kernel emulator")
except ImportError:
    try:
        from parallel_chip8 import ParallelChip8Emulator
        print("Using standard parallel emulator")
    except ImportError:
        print("Error: No emulator available")
        exit(1)


def load_rom_file(filepath: str) -> bytes:
    """Load a ROM file from disk"""
    with open(filepath, 'rb') as f:
        return f.read()


def test_single_rom(rom_path: str, num_instances: int = 1, cycles: int = 5000, 
                   output_dir: str = "output", scale: int = 8, quirks: dict = None):
    """Test a single ROM on the parallel emulator"""
    print(f"Loading ROM: {rom_path}")
    
    # Load ROM data
    rom_data = load_rom_file(rom_path)
    print(f"ROM size: {len(rom_data)} bytes")
    
    # Create emulator
    print(f"Creating emulator with {num_instances} instances...")
    emulator = ParallelChip8Emulator(num_instances, quirks=quirks)
    
    # Load ROM into all instances
    emulator.load_single_rom(rom_data)
    
    # Run emulation
    print(f"Running emulation for {cycles} cycles...")
    start_time = time.time()
    emulator.run(cycles=cycles)
    end_time = time.time()
    
    # Print results
    print(f"Emulation completed in {end_time - start_time:.2f} seconds")
    emulator.print_aggregate_stats()
    
    # Save display outputs
    rom_name = Path(rom_path).stem
    instance_output_dir = os.path.join(output_dir, f"{rom_name}_instances")
    saved_files = emulator.save_displays_as_pngs(
        instance_output_dir, 
        scale=scale, 
        prefix=f"{rom_name}"
    )
    
    print(f"Display outputs saved to: {instance_output_dir}")
    return emulator, saved_files


def test_multiple_roms(rom_paths: list, num_instances_per_rom: int = 1, cycles: int = 5000,
                      output_dir: str = "output", scale: int = 8, quirks: dict = None):
    """Test multiple ROMs, one instance per ROM"""
    print(f"Testing {len(rom_paths)} ROMs with {num_instances_per_rom} instances each...")
    
    all_results = []
    
    for rom_path in rom_paths:
        print(f"\n{'='*60}")
        try:
            emulator, saved_files = test_single_rom(
                rom_path, num_instances_per_rom, cycles, output_dir, scale, quirks
            )
            all_results.append({
                'rom_path': rom_path,
                'emulator': emulator,
                'saved_files': saved_files,
                'success': True
            })
        except Exception as e:
            print(f"ERROR testing {rom_path}: {e}")
            all_results.append({
                'rom_path': rom_path,
                'error': str(e),
                'success': False
            })
    
    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY:")
    successful = sum(1 for r in all_results if r['success'])
    print(f"Successfully tested: {successful}/{len(rom_paths)} ROMs")
    
    for result in all_results:
        status = "✓" if result['success'] else "✗"
        print(f"  {status} {Path(result['rom_path']).name}")
    
    return all_results


def test_parallel_roms(rom_paths: list, total_instances: int, cycles: int = 5000,
                      output_dir: str = "output", scale: int = 8, quirks: dict = None):
    """Test multiple ROMs in parallel on a single emulator instance"""
    print(f"Testing {len(rom_paths)} ROMs in parallel with {total_instances} total instances...")
    
    # Load all ROM data
    rom_data_list = []
    for rom_path in rom_paths:
        rom_data = load_rom_file(rom_path)
        rom_data_list.append(rom_data)
        print(f"Loaded {Path(rom_path).name}: {len(rom_data)} bytes")
    
    # Create emulator
    print(f"Creating emulator with {total_instances} instances...")
    emulator = ParallelChip8Emulator(total_instances, quirks=quirks)
    
    # Load ROMs (will cycle through if fewer ROMs than instances)
    emulator.load_roms(rom_data_list)
    
    # Run emulation
    print(f"Running parallel emulation for {cycles} cycles...")
    start_time = time.time()
    emulator.run(cycles=cycles)
    end_time = time.time()
    
    # Print results
    print(f"Parallel emulation completed in {end_time - start_time:.2f} seconds")
    emulator.print_aggregate_stats()
    
    # Save display outputs for all instances
    parallel_output_dir = os.path.join(output_dir, "parallel_test")
    saved_files = emulator.save_displays_as_pngs(
        parallel_output_dir,
        scale=scale,
        prefix="parallel"
    )
    
    print(f"All display outputs saved to: {parallel_output_dir}")
    return emulator, saved_files


def find_test_roms(test_dir: str = "chip8-test-suite/bin") -> list:
    """Find CHIP-8 test ROMs in the specified directory"""
    possible_paths = [
        test_dir,
        "chip8-test-suite/bin",
        "../chip8-test-suite/bin", 
        "../../chip8-test-suite/bin",
        os.path.join(os.path.dirname(__file__), "chip8-test-suite", "bin")
    ]
    
    for path in possible_paths:
        if os.path.exists(path):
            test_files = glob.glob(os.path.join(path, "*.ch8"))
            if test_files:
                print(f"Found test directory: {os.path.abspath(path)}")
                return sorted(test_files)
    
    print("No test ROMs found. Download with:")
    print("  git clone https://github.com/Timendus/chip8-test-suite.git")
    return []


def main():
    parser = argparse.ArgumentParser(description="Test parallel CHIP-8 emulator")
    parser.add_argument("--rom", type=str, help="Single ROM file to test")
    parser.add_argument("--test-suite", action="store_true", help="Run CHIP-8 test suite")
    parser.add_argument("--parallel", action="store_true", help="Run ROMs in parallel mode")
    parser.add_argument("--instances", type=int, default=1, help="Number of instances")
    parser.add_argument("--cycles", type=int, default=5000, help="Number of cycles to run")
    parser.add_argument("--output", type=str, default="output", help="Output directory")
    parser.add_argument("--scale", type=int, default=8, help="Display scale factor")
    parser.add_argument("--test-dir", type=str, default="chip8-test-suite/bin", 
                       help="Test suite directory")
    
    # Quirks options
    parser.add_argument("--no-memory-quirk", action="store_true", 
                       help="Disable memory quirk (Fx55/Fx65 don't increment I)")
    parser.add_argument("--display-wait", action="store_true",
                       help="Enable display wait quirk (slower but more accurate)")
    parser.add_argument("--no-jumping-quirk", action="store_true",
                       help="Disable jumping quirk (Bnnn uses V0 instead of VX)")
    parser.add_argument("--shifting-quirk", action="store_true",
                       help="Enable shifting quirk (8xy6/8xyE use VY instead of VX)")
    parser.add_argument("--no-logic-quirk", action="store_true",
                       help="Disable logic quirk (8xy1/8xy2/8xy3 don't reset VF)")
    
    args = parser.parse_args()
    
    # Configure quirks
    quirks = {
        'memory': not args.no_memory_quirk,
        'display_wait': args.display_wait,
        'jumping': not args.no_jumping_quirk,
        'shifting': args.shifting_quirk,
        'logic': not args.no_logic_quirk,
    }
    
    print("CHIP-8 Parallel Emulator Test Runner")
    print("=" * 50)
    print(f"Quirks configuration:")
    for quirk, enabled in quirks.items():
        print(f"  {quirk}: {'ON' if enabled else 'OFF'}")
    print()
    
    # Create output directory
    os.makedirs(args.output, exist_ok=True)
    
    if args.rom:
        # Test single ROM
        if not os.path.exists(args.rom):
            print(f"ROM file not found: {args.rom}")
            return 1
        
        test_single_rom(
            args.rom, 
            num_instances=args.instances,
            cycles=args.cycles,
            output_dir=args.output,
            scale=args.scale,
            quirks=quirks
        )
        
    elif args.test_suite:
        # Test the CHIP-8 test suite
        test_roms = find_test_roms(args.test_dir)
        if not test_roms:
            return 1
        
        print(f"Found {len(test_roms)} test ROMs")
        for rom in test_roms:
            print(f"  - {Path(rom).name}")
        print()
        
        if args.parallel:
            # Run all ROMs in parallel
            test_parallel_roms(
                test_roms,
                total_instances=args.instances,
                cycles=args.cycles,
                output_dir=args.output,
                scale=args.scale,
                quirks=quirks
            )
        else:
            # Run each ROM separately
            test_multiple_roms(
                test_roms,
                num_instances_per_rom=args.instances,
                cycles=args.cycles,
                output_dir=args.output,
                scale=args.scale,
                quirks=quirks
            )
    
    else:
        # Interactive mode
        print("Interactive mode - choose an option:")
        print("1. Test single ROM file")
        print("2. Test CHIP-8 test suite (separate)")
        print("3. Test CHIP-8 test suite (parallel)")
        print("4. Performance test (1000 instances, simple ROM)")
        
        choice = input("Enter choice (1-4): ").strip()
        
        if choice == "1":
            rom_path = input("Enter ROM path: ").strip()
            if os.path.exists(rom_path):
                test_single_rom(
                    rom_path,
                    num_instances=args.instances,
                    cycles=args.cycles,
                    output_dir=args.output,
                    scale=args.scale,
                    quirks=quirks
                )
            else:
                print(f"ROM file not found: {rom_path}")
        
        elif choice == "2":
            test_roms = find_test_roms(args.test_dir)
            if test_roms:
                test_multiple_roms(
                    test_roms,
                    num_instances_per_rom=args.instances,
                    cycles=args.cycles,
                    output_dir=args.output,
                    scale=args.scale,
                    quirks=quirks
                )
        
        elif choice == "3":
            test_roms = find_test_roms(args.test_dir)
            if test_roms:
                test_parallel_roms(
                    test_roms,
                    total_instances=args.instances,
                    cycles=args.cycles,
                    output_dir=args.output,
                    scale=args.scale,
                    quirks=quirks
                )
        
        elif choice == "4":
            # Performance test with simple ROM
            print("Creating simple test ROM for performance testing...")
            # Simple ROM that draws a pattern
            test_rom = bytes([
                0xA2, 0x10,  # LD I, 0x210 (sprite data location)
                0x60, 0x10,  # LD V0, 16 (x position)
                0x61, 0x10,  # LD V1, 16 (y position)
                0xD0, 0x15,  # DRW V0, V1, 5 (draw sprite)
                0x70, 0x01,  # ADD V0, 1 (move x position)
                0x30, 0x3F,  # SE V0, 63 (check if at right edge)
                0x12, 0x08,  # JP 0x208 (jump to ADD instruction)
                0x60, 0x00,  # LD V0, 0 (reset x position)
                0x12, 0x08,  # JP 0x208 (continue loop)
                # Sprite data at 0x210
                0xF0, 0x90, 0x90, 0x90, 0xF0  # Simple square pattern
            ])
            
            print("Running performance test with 1000 instances...")
            emulator = ParallelChip8Emulator(1000, quirks=quirks)
            emulator.load_single_rom(test_rom)
            
            start_time = time.time()
            emulator.run(max_cycles=10000)
            end_time = time.time()
            
            print(f"Performance test completed in {end_time - start_time:.2f} seconds")
            emulator.print_aggregate_stats()
            
            # Save a sample of displays
            sample_ids = list(range(0, 1000, 100))  # Every 100th instance
            perf_output_dir = os.path.join(args.output, "performance_test")
            emulator.save_displays_as_pngs(
                perf_output_dir,
                instance_ids=sample_ids,
                scale=args.scale,
                prefix="perf_sample"
            )
            print(f"Sample displays saved to: {perf_output_dir}")
        
        else:
            print("Invalid choice")
            return 1
    
    print("\nTest completed!")
    return 0


if __name__ == "__main__":
    exit(main())