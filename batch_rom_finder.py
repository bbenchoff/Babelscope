#!/usr/bin/env python3
"""
Test Random CHIP-8 ROMs with Integrated CA Detection
Loads random ROM data and tests which ones actually execute without crashing
Now includes real-time cellular automata detection for computational archaeology
"""

import os
import sys
import glob
import argparse
import time
import numpy as np
import cupy as cp
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Set
from collections import defaultdict, Counter
from dataclasses import dataclass

# Add the emulators and generators directories to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'emulators'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'generators'))

try:
    from mega_kernel_chip8 import MegaKernelChip8Emulator as ParallelChip8Emulator
    print("Using mega-kernel emulator")
except ImportError:
    try:
        from parallel_chip8 import ParallelChip8Emulator
        print("Using standard parallel emulator")
    except ImportError:
        print("Error: No emulator available")
        sys.exit(1)

# Also import single-instance emulator for CA analysis
try:
    from chip8 import Chip8Emulator
    print("Single-instance emulator available for CA analysis")
except ImportError:
    print("Warning: Single-instance emulator not available - CA detection disabled")
    Chip8Emulator = None

try:
    from random_chip8_generator import PureRandomChip8Generator
    print("Pure random ROM generator available")
except ImportError:
    print("Warning: Pure random ROM generator not available")
    PureRandomChip8Generator = None


@dataclass
class CAAnalysis:
    """Results of cellular automata analysis"""
    ca_likelihood: float  # 0-100%
    memory_sequential: bool
    state_evolution: bool
    display_patterns: bool
    neighbor_checking: bool
    rule_complexity: str  # "simple", "moderate", "complex"
    classification: str
    evidence: List[str]
    hot_loop_pc_range: Tuple[int, int]
    execution_percentage: float


class QuickCADetector:
    """Fast CA detector optimized for real-time analysis of random ROMs"""
    
    def __init__(self):
        self.instruction_patterns = {
            # Memory operations
            'LD_I': 0xA000,      # LD I, addr
            'ADD_I': 0xF01E,     # ADD I, Vx  
            'LD_MEM_STORE': 0xF055,  # LD [I], Vx
            'LD_MEM_LOAD': 0xF065,   # LD Vx, [I]
            
            # Logic operations
            'OR': 0x8001,        # OR Vx, Vy
            'AND': 0x8002,       # AND Vx, Vy
            'XOR': 0x8003,       # XOR Vx, Vy
            'ADD_REG': 0x8004,   # ADD Vx, Vy
            
            # Display
            'DRW': 0xD000,       # DRW Vx, Vy, n
            
            # Control flow
            'JP': 0x1000,        # JP addr
            'CALL': 0x2000,      # CALL addr
        }
    
    def analyze_rom_for_ca(self, rom_data: bytes, max_cycles: int = 50000) -> Optional[CAAnalysis]:
        """Quick CA analysis of a ROM using single-instance emulator"""
        if Chip8Emulator is None:
            return None
        
        try:
            # Create emulator and load ROM
            emulator = Chip8Emulator()
            emulator.load_rom(rom_data)
            
            # Track execution patterns
            pc_frequency = defaultdict(int)
            total_cycles = 0
            
            # Run ROM and track PC frequency
            for cycle in range(max_cycles):
                if emulator.crashed or emulator.halt:
                    break
                
                pc = emulator.program_counter
                pc_frequency[pc] += 1
                
                if not emulator.step():
                    break
                
                total_cycles += 1
                
                # Early exit if stuck in infinite loop with no display changes
                if cycle > 10000 and emulator.stats.get('display_writes', 0) == 0:
                    break
            
            if total_cycles < 1000:  # Too short to be interesting
                return None
            
            # Find hot execution region
            if not pc_frequency:
                return None
            
            sorted_pcs = sorted(pc_frequency.items(), key=lambda x: x[1], reverse=True)
            total_executions = sum(pc_frequency.values())
            
            # Find the most executed region
            hot_pc_start = sorted_pcs[0][0]
            hot_pc_end = hot_pc_start
            hot_execution_count = sorted_pcs[0][1]
            
            # Expand region to include nearby frequently executed addresses
            for pc, count in sorted_pcs[1:10]:  # Check top 10 most frequent
                if abs(pc - hot_pc_start) <= 20:  # Within 20 bytes
                    hot_pc_start = min(hot_pc_start, pc)
                    hot_pc_end = max(hot_pc_end, pc)
                    hot_execution_count += count
            
            execution_percentage = hot_execution_count / total_executions * 100
            
            # Only analyze if hot region is significant
            if execution_percentage < 50:
                return None
            
            # Analyze the hot region for CA patterns
            ca_score = 0.0
            evidence = []
            
            memory_sequential = self._check_memory_patterns(emulator.memory, hot_pc_start, hot_pc_end)
            state_evolution = self._check_state_evolution_patterns(emulator.memory, hot_pc_start, hot_pc_end)
            display_patterns = emulator.stats.get('display_writes', 0) > 0 and execution_percentage > 70
            neighbor_checking = self._check_neighbor_patterns(emulator.memory, hot_pc_start, hot_pc_end)
            
            # Score CA likelihood
            if memory_sequential:
                ca_score += 25
                evidence.append("Sequential memory access patterns")
            
            if state_evolution:
                ca_score += 30
                evidence.append("Read-modify-write state evolution")
            
            if display_patterns:
                ca_score += 20
                evidence.append("Structured display updates in tight loop")
            
            if neighbor_checking:
                ca_score += 25
                evidence.append("Neighbor-checking memory access")
            
            # Check for specific CA-like instruction sequences
            xor_and_display = self._check_xor_display_pattern(emulator.memory, hot_pc_start, hot_pc_end)
            if xor_and_display:
                ca_score += 15
                evidence.append("XOR logic with display output")
            
            # Only return analysis if CA likelihood is significant
            if ca_score < 40:
                return None
            
            # Classify the pattern
            classification = self._classify_ca_pattern(emulator.memory, hot_pc_start, hot_pc_end, 
                                                    memory_sequential, state_evolution)
            
            rule_complexity = self._assess_rule_complexity(emulator.memory, hot_pc_start, hot_pc_end)
            
            return CAAnalysis(
                ca_likelihood=min(ca_score, 100.0),
                memory_sequential=memory_sequential,
                state_evolution=state_evolution,
                display_patterns=display_patterns,
                neighbor_checking=neighbor_checking,
                rule_complexity=rule_complexity,
                classification=classification,
                evidence=evidence,
                hot_loop_pc_range=(hot_pc_start, hot_pc_end),
                execution_percentage=execution_percentage
            )
            
        except Exception as e:
            # Silently fail for individual ROM analysis
            return None
    
    def _check_memory_patterns(self, memory: np.ndarray, start: int, end: int) -> bool:
        """Check for sequential memory access patterns"""
        has_add_i = False
        has_memory_op = False
        
        for addr in range(start, min(end + 1, len(memory) - 1), 2):
            instruction = (int(memory[addr]) << 8) | int(memory[addr + 1])
            
            # Check for ADD I, Vx pattern
            if (instruction & 0xF0FF) == 0xF01E:
                has_add_i = True
            
            # Check for memory operations
            if (instruction & 0xF0FF) in [0xF055, 0xF065]:  # LD [I], Vx or LD Vx, [I]
                has_memory_op = True
        
        return has_add_i and has_memory_op
    
    def _check_state_evolution_patterns(self, memory: np.ndarray, start: int, end: int) -> bool:
        """Check for read-modify-write patterns"""
        has_read = False
        has_modify = False
        has_write = False
        
        for addr in range(start, min(end + 1, len(memory) - 1), 2):
            instruction = (int(memory[addr]) << 8) | int(memory[addr + 1])
            opcode = (instruction & 0xF000) >> 12
            
            # Memory read
            if (instruction & 0xF0FF) == 0xF065:
                has_read = True
            
            # Logic/arithmetic operations
            if opcode == 0x8 and (instruction & 0x000F) in [0x1, 0x2, 0x3, 0x4, 0x5]:
                has_modify = True
            
            # Memory write
            if (instruction & 0xF0FF) == 0xF055:
                has_write = True
        
        return has_read and has_modify and has_write
    
    def _check_neighbor_patterns(self, memory: np.ndarray, start: int, end: int) -> bool:
        """Check for patterns that suggest neighbor checking"""
        add_i_count = 0
        memory_read_count = 0
        
        for addr in range(start, min(end + 1, len(memory) - 1), 2):
            instruction = (int(memory[addr]) << 8) | int(memory[addr + 1])
            
            if (instruction & 0xF0FF) == 0xF01E:  # ADD I, Vx
                add_i_count += 1
            
            if (instruction & 0xF0FF) == 0xF065:  # LD Vx, [I]
                memory_read_count += 1
        
        # Multiple index manipulations + memory reads suggests neighbor checking
        return add_i_count >= 2 and memory_read_count >= 2
    
    def _check_xor_display_pattern(self, memory: np.ndarray, start: int, end: int) -> bool:
        """Check for XOR operations combined with display output"""
        has_xor = False
        has_display = False
        
        for addr in range(start, min(end + 1, len(memory) - 1), 2):
            instruction = (int(memory[addr]) << 8) | int(memory[addr + 1])
            opcode = (instruction & 0xF000) >> 12
            
            if opcode == 0x8 and (instruction & 0x000F) == 0x3:  # XOR
                has_xor = True
            
            if opcode == 0xD:  # DRW
                has_display = True
        
        return has_xor and has_display
    
    def _classify_ca_pattern(self, memory: np.ndarray, start: int, end: int, 
                           sequential: bool, evolution: bool) -> str:
        """Classify the type of CA pattern"""
        if not (sequential or evolution):
            return "Non-CA computational pattern"
        
        has_xor = False
        has_display = False
        
        for addr in range(start, min(end + 1, len(memory) - 1), 2):
            instruction = (int(memory[addr]) << 8) | int(memory[addr + 1])
            opcode = (instruction & 0xF000) >> 12
            
            if opcode == 0x8 and (instruction & 0x000F) == 0x3:
                has_xor = True
            if opcode == 0xD:
                has_display = True
        
        if has_xor and has_display and evolution:
            return "XOR-based visual cellular automaton"
        elif sequential and evolution and has_display:
            return "Memory-grid cellular automaton"
        elif sequential and has_display:
            return "Linear scanning automaton"
        elif evolution:
            return "State evolution pattern"
        else:
            return "Memory processing pattern"
    
    def _assess_rule_complexity(self, memory: np.ndarray, start: int, end: int) -> str:
        """Assess complexity of the computational rule"""
        logical_ops = 0
        arithmetic_ops = 0
        
        for addr in range(start, min(end + 1, len(memory) - 1), 2):
            instruction = (int(memory[addr]) << 8) | int(memory[addr + 1])
            opcode = (instruction & 0xF000) >> 12
            
            if opcode == 0x8:
                subop = instruction & 0x000F
                if subop in [0x1, 0x2, 0x3]:  # OR, AND, XOR
                    logical_ops += 1
                elif subop in [0x4, 0x5, 0x7]:  # ADD, SUB, SUBN
                    arithmetic_ops += 1
        
        total_ops = logical_ops + arithmetic_ops
        
        if total_ops <= 1:
            return "simple"
        elif total_ops <= 3:
            return "moderate"
        else:
            return "complex"


def load_rom_files(rom_dir: str) -> List[Tuple[str, bytes]]:
    """Load all ROM files from a directory"""
    rom_files = []
    
    # Look for .ch8 and .bin files
    for pattern in ["*.ch8", "*.bin"]:
        files = glob.glob(os.path.join(rom_dir, pattern))
        for filepath in files:
            try:
                with open(filepath, 'rb') as f:
                    rom_data = f.read()
                rom_files.append((os.path.basename(filepath), rom_data))
            except Exception as e:
                print(f"Error loading {filepath}: {e}")
    
    return sorted(rom_files)


def test_rom_batch(rom_data_list: List[bytes], cycles: int = 1000000, 
                  ca_detector: Optional[QuickCADetector] = None) -> Tuple[List[Dict], List[Tuple[bytes, CAAnalysis]]]:
    """
    Test a batch of ROMs in parallel and detect CA patterns
    Returns (results, ca_roms) where ca_roms contains ROM data and CA analysis for CA-like ROMs
    """
    print(f"Testing batch of {len(rom_data_list)} ROMs...")
    
    # Create emulator for the batch
    emulator = ParallelChip8Emulator(len(rom_data_list))
    
    # Load all ROMs
    emulator.load_roms(rom_data_list)
    
    # Run all ROMs
    start_time = time.time()
    emulator.run(cycles=cycles)
    execution_time = time.time() - start_time
    
    print(f"Batch execution completed in {execution_time:.2f} seconds")
    
    # Analyze results for each ROM
    results = []
    displays = emulator.get_displays()
    ca_roms = []  # Store ROMs with CA patterns
    
    for i in range(len(rom_data_list)):
        # Get state for this instance
        final_pc = int(emulator.program_counter[i])
        crashed = bool(emulator.crashed[i])
        halted = bool(emulator.halted[i])
        waiting_for_key = bool(emulator.waiting_for_key[i])
        
        # Analyze display
        display = displays[i]
        pixels_set = int(np.sum(display > 0))
        total_pixels = display.shape[0] * display.shape[1]
        pixel_density = pixels_set / total_pixels
        
        has_output = pixels_set > 0
        has_structure = pixel_density > 0.05 and pixel_density < 0.5
        
        # Get per-instance stats
        instructions = int(emulator.stats['instructions_executed'][i])
        display_writes = int(emulator.stats['display_writes'][i])
        pixels_drawn = int(emulator.stats['pixels_drawn'][i])
        
        # Check if ROM is potentially interesting for CA analysis
        potentially_ca = (not crashed and not waiting_for_key and 
                         has_output and instructions > 1000 and display_writes > 0)
        
        result = {
            'execution_time': execution_time / len(rom_data_list),  # Approximate
            'final_pc': final_pc,
            'crashed': crashed,
            'halted': halted,
            'waiting_for_key': waiting_for_key,
            'completed_normally': not crashed and not waiting_for_key,
            'instructions_executed': instructions,
            'display_writes': display_writes,
            'pixels_drawn': pixels_drawn,
            'final_pixel_count': pixels_set,
            'pixel_density': pixel_density,
            'has_output': has_output,
            'has_structure': has_structure,
            'interesting': (not crashed and not waiting_for_key) and (has_output and has_structure),
            'potentially_ca': potentially_ca,
            'ca_analysis': None  # Will be filled if CA detected
        }
        
        # Run CA detection on potentially interesting ROMs
        if potentially_ca and ca_detector is not None:
            ca_analysis = ca_detector.analyze_rom_for_ca(rom_data_list[i], max_cycles=cycles//10)
            if ca_analysis is not None and ca_analysis.ca_likelihood >= 40:
                result['ca_analysis'] = ca_analysis
                ca_roms.append((rom_data_list[i], ca_analysis))
                print(f"  CA DETECTED! Likelihood: {ca_analysis.ca_likelihood:.0f}% - {ca_analysis.classification}")
        
        results.append(result)
    
    return results, ca_roms


def save_interesting_roms(rom_files: List[Tuple[str, bytes]], results: List[Dict], 
                         output_dir: str = "interesting_roms"):
    """Save ROMs that produced interesting results and generate screenshots"""
    os.makedirs(output_dir, exist_ok=True)
    
    saved_count = 0
    ca_count = 0
    
    for (filename, rom_data), result in zip(rom_files, results):
        if result['interesting'] or result.get('ca_analysis') is not None:
            # Create descriptive filename
            base_name = Path(filename).stem
            
            # Add CA marker if this is a CA ROM
            ca_suffix = ""
            if result.get('ca_analysis') is not None:
                ca_analysis = result['ca_analysis']
                ca_suffix = f"_CA{ca_analysis.ca_likelihood:.0f}"
                ca_count += 1
            
            new_filename = (f"{base_name}"
                          f"_inst{result['instructions_executed']}"
                          f"_pix{result['final_pixel_count']}"
                          f"_dens{result['pixel_density']:.3f}"
                          f"{ca_suffix}")
            
            # Save ROM file
            rom_path = os.path.join(output_dir, f"{new_filename}.ch8")
            with open(rom_path, 'wb') as f:
                f.write(rom_data)
            
            # Generate screenshot by running the ROM again
            print(f"Generating screenshot for {new_filename}...")
            try:
                # Create single-instance emulator
                emulator = ParallelChip8Emulator(1)
                emulator.load_single_rom(rom_data)
                
                # Run for the same number of cycles as the original test
                emulator.run(cycles=1000000)
                
                # Get display and convert to numpy
                display = emulator.get_displays()[0]  # Shape: (32, 64)
                if hasattr(display, 'get'):  # CuPy array
                    display_np = display.get()
                else:  # Already numpy
                    display_np = display
                
                # Convert to PIL Image
                from PIL import Image
                display_img = (display_np * 255).astype(np.uint8)
                
                # Scale up 8x for visibility (256x512 pixels)
                scale_factor = 8
                scaled_img = np.repeat(np.repeat(display_img, scale_factor, axis=0), scale_factor, axis=1)
                
                # Save as PNG
                img = Image.fromarray(scaled_img, mode='L')
                screenshot_path = os.path.join(output_dir, f"{new_filename}.png")
                img.save(screenshot_path)
                
                # Save CA analysis if available
                if result.get('ca_analysis') is not None:
                    ca_report_path = os.path.join(output_dir, f"{new_filename}_CA_ANALYSIS.txt")
                    with open(ca_report_path, 'w') as f:
                        ca = result['ca_analysis']
                        f.write(f"Cellular Automata Analysis\n")
                        f.write(f"=" * 40 + "\n\n")
                        f.write(f"ROM: {new_filename}.ch8\n")
                        f.write(f"CA Likelihood: {ca.ca_likelihood:.1f}%\n")
                        f.write(f"Classification: {ca.classification}\n")
                        f.write(f"Rule Complexity: {ca.rule_complexity}\n")
                        f.write(f"Hot Loop: 0x{ca.hot_loop_pc_range[0]:03X}-0x{ca.hot_loop_pc_range[1]:03X}\n")
                        f.write(f"Execution %: {ca.execution_percentage:.1f}%\n\n")
                        f.write(f"Pattern Features:\n")
                        f.write(f"- Memory Sequential: {'Yes' if ca.memory_sequential else 'No'}\n")
                        f.write(f"- State Evolution: {'Yes' if ca.state_evolution else 'No'}\n")
                        f.write(f"- Display Patterns: {'Yes' if ca.display_patterns else 'No'}\n")
                        f.write(f"- Neighbor Checking: {'Yes' if ca.neighbor_checking else 'No'}\n\n")
                        f.write(f"Evidence:\n")
                        for evidence in ca.evidence:
                            f.write(f"- {evidence}\n")
                
            except Exception as e:
                print(f"Error generating screenshot for {new_filename}: {e}")
            
            saved_count += 1
    
    print(f"Saved {saved_count} interesting ROMs ({ca_count} with CA patterns) to {output_dir}/")
    return saved_count, ca_count


def save_ca_roms_only(ca_roms: List[Tuple[bytes, CAAnalysis]], output_dir: str = "ca_roms"):
    """Save only ROMs with CA patterns to a dedicated directory"""
    os.makedirs(output_dir, exist_ok=True)
    
    for i, (rom_data, ca_analysis) in enumerate(ca_roms):
        filename = f"ca_rom_{i:06d}_likelihood{ca_analysis.ca_likelihood:.0f}_{ca_analysis.classification.replace(' ', '_')}"
        
        # Save ROM
        rom_path = os.path.join(output_dir, f"{filename}.ch8")
        with open(rom_path, 'wb') as f:
            f.write(rom_data)
        
        # Save analysis
        analysis_path = os.path.join(output_dir, f"{filename}_analysis.txt")
        with open(analysis_path, 'w') as f:
            f.write(f"Cellular Automata Analysis\n")
            f.write(f"=" * 40 + "\n\n")
            f.write(f"CA Likelihood: {ca_analysis.ca_likelihood:.1f}%\n")
            f.write(f"Classification: {ca_analysis.classification}\n")
            f.write(f"Rule Complexity: {ca_analysis.rule_complexity}\n")
            f.write(f"Hot Loop: 0x{ca_analysis.hot_loop_pc_range[0]:03X}-0x{ca_analysis.hot_loop_pc_range[1]:03X}\n")
            f.write(f"Execution %: {ca_analysis.execution_percentage:.1f}%\n\n")
            f.write(f"Pattern Features:\n")
            f.write(f"- Memory Sequential: {'Yes' if ca_analysis.memory_sequential else 'No'}\n")
            f.write(f"- State Evolution: {'Yes' if ca_analysis.state_evolution else 'No'}\n")
            f.write(f"- Display Patterns: {'Yes' if ca_analysis.display_patterns else 'No'}\n")
            f.write(f"- Neighbor Checking: {'Yes' if ca_analysis.neighbor_checking else 'No'}\n\n")
            f.write(f"Evidence:\n")
            for evidence in ca_analysis.evidence:
                f.write(f"- {evidence}\n")
    
    print(f"Saved {len(ca_roms)} CA ROMs to {output_dir}/")
    return len(ca_roms)


def generate_and_test_random_batch(generator, num_roms: int = 10000, 
                                 test_cycles: int = 1000000,
                                 ca_detector: Optional[QuickCADetector] = None) -> Tuple[int, int, int]:
    """Generate pure random ROMs and test them immediately, returning (completed, interesting, ca_count)"""
    print(f"Generating and testing {num_roms:,} pure random ROMs...")
    
    # Generate pure random ROMs - no filtering at all
    roms = generator.generate_batch(num_roms)
    
    # Convert to list of bytes
    rom_data_list = [cp.asnumpy(rom).tobytes() for rom in roms]
    
    # Test them all with CA detection
    results, ca_roms = test_rom_batch(rom_data_list, cycles=test_cycles, ca_detector=ca_detector)
    
    # Count results
    interesting_count = sum(1 for r in results if r['interesting'])
    completed_count = sum(1 for r in results if r['completed_normally'])
    has_output_count = sum(1 for r in results if r['has_output'])
    crashed_count = sum(1 for r in results if r['crashed'])
    ca_count = len(ca_roms)
    
    print(f"Results: {crashed_count} crashed, {completed_count} completed normally, "
          f"{has_output_count} had visual output, {interesting_count} were interesting, "
          f"{ca_count} had CA patterns")
    
    # Save ROMs if there are interesting ones
    if interesting_count > 0 or ca_count > 0:
        # Create fake filenames for the ROM data
        rom_files = [(f"random_{i:06d}.ch8", rom_data) for i, rom_data in enumerate(rom_data_list)]
        save_interesting_roms(rom_files, results, "output/interesting_roms")
    
    # Save CA ROMs to dedicated directory
    if ca_count > 0:
        save_ca_roms_only(ca_roms, "output/ca_roms")
    
    return completed_count, interesting_count, ca_count


def print_summary_stats(results: List[Dict]):
    """Print summary statistics of ROM testing"""
    total = len(results)
    crashed = sum(1 for r in results if r['crashed'])
    completed = sum(1 for r in results if r['completed_normally'])
    has_output = sum(1 for r in results if r['has_output'])
    interesting = sum(1 for r in results if r['interesting'])
    ca_roms = sum(1 for r in results if r.get('ca_analysis') is not None)
    
    print("\nSummary Statistics:")
    print("=" * 50)
    print(f"Total ROMs tested:      {total:,}")
    print(f"Crashed:                {crashed:,} ({crashed/total*100:.1f}%)")
    print(f"Completed normally:     {completed:,} ({completed/total*100:.1f}%)")
    print(f"Produced visual output: {has_output:,} ({has_output/total*100:.1f}%)")
    print(f"Interesting:            {interesting:,} ({interesting/total*100:.1f}%)")
    print(f"CA patterns detected:   {ca_roms:,} ({ca_roms/total*100:.4f}%)")
    
    if completed > 0:
        avg_instructions = np.mean([r['instructions_executed'] for r in results if r['completed_normally']])
        print(f"Avg instructions (completed): {avg_instructions:,.0f}")
    
    if has_output > 0:
        avg_pixels = np.mean([r['final_pixel_count'] for r in results if r['has_output']])
        print(f"Avg pixels drawn (visual): {avg_pixels:.0f}")
    
    if ca_roms > 0:
        ca_likelihoods = [r['ca_analysis'].ca_likelihood for r in results if r.get('ca_analysis')]
        avg_ca_likelihood = np.mean(ca_likelihoods)
        max_ca_likelihood = np.max(ca_likelihoods)
        print(f"CA likelihood (avg/max): {avg_ca_likelihood:.1f}%/{max_ca_likelihood:.1f}%")


def main():
    parser = argparse.ArgumentParser(description="Test random CHIP-8 ROMs with CA detection")
    parser.add_argument("--rom-dir", type=str, help="Directory containing ROM files to test")
    parser.add_argument("--generate", type=int, help="Generate N random ROMs and test them")
    parser.add_argument("--continuous", action="store_true", help="Run continuously in batches")
    parser.add_argument("--cycles", type=int, default=1000000, help="Cycles to run each ROM")
    parser.add_argument("--batch-size", type=int, default=10000, help="Batch size for continuous mode")
    parser.add_argument("--output", type=str, default="output", help="Output directory")
    parser.add_argument("--save-interesting", action="store_true", help="Save interesting ROMs")
    parser.add_argument("--ca-only", action="store_true", help="Only save ROMs with CA patterns")
    parser.add_argument("--no-ca", action="store_true", help="Disable CA detection (faster)")
    
    args = parser.parse_args()
    
    print("Random CHIP-8 ROM Tester with CA Detection - Computational Archaeology")
    print("=" * 70)
    
    os.makedirs(args.output, exist_ok=True)
    
    # Initialize CA detector unless disabled
    ca_detector = None if args.no_ca else QuickCADetector()
    if ca_detector:
        print("CA detection enabled - will analyze ROMs for cellular automata patterns")
    else:
        print("CA detection disabled")
    
    if args.continuous:
        # Continuous mode - run forever in batches
        if PureRandomChip8Generator is None:
            print("Error: Pure random ROM generator not available")
            return 1
        
        print(f"Running continuously with batches of {args.batch_size:,} ROMs")
        print("Press Ctrl+C to stop")
        print("=" * 70)
        
        generator = PureRandomChip8Generator(rom_size=3584)
        
        completed_total = 0
        interesting_total = 0
        ca_total = 0
        batches_tested = 0
        start_time = time.time()
        
        try:
            while True:  # Run forever
                batch_start = time.time()
                
                completed, interesting, ca_count = generate_and_test_random_batch(
                    generator, args.batch_size, args.cycles, ca_detector
                )
                
                completed_total += completed
                interesting_total += interesting
                ca_total += ca_count
                batches_tested += 1
                
                batch_time = time.time() - batch_start
                total_time = time.time() - start_time
                total_roms = batches_tested * args.batch_size
                
                print(f"Batch {batches_tested}: {completed}/{args.batch_size} completed, "
                      f"{interesting} interesting, {ca_count} CA patterns ({batch_time:.1f}s)")
                print(f"Running totals: {total_roms:,} tested, {completed_total} completed, "
                      f"{interesting_total} interesting, {ca_total} CA patterns")
                
                # Show rate statistics
                rate = total_roms / total_time if total_time > 0 else 0
                success_rate = completed_total / total_roms * 100 if total_roms > 0 else 0
                interesting_rate = interesting_total / total_roms * 100 if total_roms > 0 else 0
                ca_rate = ca_total / total_roms * 100 if total_roms > 0 else 0
                
                print(f"Rates: {rate:.0f} ROMs/sec, {success_rate:.3f}% completion, "
                      f"{interesting_rate:.4f}% interesting, {ca_rate:.5f}% CA")
                print("=" * 70)
                
        except KeyboardInterrupt:
            print(f"\nStopped after {batches_tested} batches")
            print(f"Final results:")
            print(f"Total ROMs tested: {batches_tested * args.batch_size:,}")
            print(f"Completed normally: {completed_total}")
            print(f"Interesting: {interesting_total}")
            print(f"CA patterns found: {ca_total}")
            
            total_time = time.time() - start_time
            rate = (batches_tested * args.batch_size) / total_time if total_time > 0 else 0
            print(f"Average rate: {rate:.0f} ROMs/second")
            
            if ca_total > 0:
                print(f"\nCA discovery rate: 1 in {(batches_tested * args.batch_size) // ca_total:,} ROMs")
        
    elif args.generate:
        # Original single-run mode
        if PureRandomChip8Generator is None:
            print("Error: Pure random ROM generator not available")
            return 1
        
        print(f"Generating {args.generate:,} pure random ROMs...")
        generator = PureRandomChip8Generator(rom_size=3584)
        
        completed_total = 0
        interesting_total = 0
        ca_total = 0
        batches_tested = 0
        
        try:
            total_tested = 0
            while total_tested < args.generate:
                batch_size = min(20000, args.generate - total_tested)
                completed, interesting, ca_count = generate_and_test_random_batch(
                    generator, batch_size, args.cycles, ca_detector
                )
                
                completed_total += completed
                interesting_total += interesting
                ca_total += ca_count
                batches_tested += 1
                total_tested += batch_size
                
                print(f"Batch {batches_tested}: {completed}/{batch_size} completed, "
                      f"{interesting} interesting, {ca_count} CA patterns")
                print(f"Running totals: {completed_total} completed, "
                      f"{interesting_total} interesting, {ca_total} CA patterns")
                
        except KeyboardInterrupt:
            print("\nTesting stopped by user")
        
        print(f"\nFinal results after testing {total_tested} ROMs:")
        print(f"Completed normally: {completed_total}")
        print(f"Interesting: {interesting_total}")
        print(f"CA patterns found: {ca_total}")
        
        if ca_total > 0:
            print(f"CA discovery rate: 1 in {total_tested // ca_total:,} ROMs")
        
    elif args.rom_dir:
        # Test existing ROM files
        if not os.path.exists(args.rom_dir):
            print(f"ROM directory not found: {args.rom_dir}")
            return 1
        
        rom_files = load_rom_files(args.rom_dir)
        if not rom_files:
            print(f"No ROM files found in {args.rom_dir}")
            return 1
        
        print(f"Found {len(rom_files)} ROM files")
        
        all_results = []
        all_ca_roms = []
        rom_data_list = [rom_data for _, rom_data in rom_files]
        
        for i in range(0, len(rom_data_list), args.batch_size):
            end_idx = min(i + args.batch_size, len(rom_data_list))
            batch_data = rom_data_list[i:end_idx]
            
            print(f"Testing batch {i//args.batch_size + 1} "
                  f"(ROMs {i+1}-{end_idx})...")
            
            batch_results, batch_ca_roms = test_rom_batch(batch_data, cycles=args.cycles, ca_detector=ca_detector)
            all_results.extend(batch_results)
            all_ca_roms.extend(batch_ca_roms)
        
        print_summary_stats(all_results)
        
        if args.save_interesting:
            interesting_output = os.path.join(args.output, "interesting_roms")
            save_interesting_roms(rom_files, all_results, interesting_output)
        
        if all_ca_roms:
            ca_output = os.path.join(args.output, "ca_roms")
            save_ca_roms_only(all_ca_roms, ca_output)
    
    else:
        # Interactive mode
        print("Options:")
        print("1. Generate and test random ROMs (single run)")
        print("2. Run continuously in batches") 
        print("3. Test existing ROM files")
        print("4. Analyze existing interesting ROMs for CA patterns")
        
        choice = input("Enter choice (1-4): ").strip()
        
        if choice == "1":
            if PureRandomChip8Generator is None:
                print("Error: Pure random ROM generator not available")
                return 1
            
            num_roms = int(input("Number of ROMs to generate (default 10000): ") or "10000")
            generator = PureRandomChip8Generator(rom_size=3584)
            completed, interesting, ca_count = generate_and_test_random_batch(
                generator, num_roms, args.cycles, ca_detector
            )
            print(f"Final: {completed} completed, {interesting} interesting, {ca_count} CA patterns")
            
        elif choice == "2":
            if PureRandomChip8Generator is None:
                print("Error: Pure random ROM generator not available")
                return 1
            
            batch_size = int(input("Batch size (default 10000): ") or "10000")
            print(f"Running continuously with batches of {batch_size:,} ROMs")
            print("Press Ctrl+C to stop")
            
            generator = PureRandomChip8Generator(rom_size=3584)
            completed_total = 0
            interesting_total = 0
            ca_total = 0
            batches_tested = 0
            start_time = time.time()
            
            try:
                while True:
                    completed, interesting, ca_count = generate_and_test_random_batch(
                        generator, batch_size, args.cycles, ca_detector
                    )
                    
                    completed_total += completed
                    interesting_total += interesting
                    ca_total += ca_count
                    batches_tested += 1
                    
                    total_time = time.time() - start_time
                    total_roms = batches_tested * batch_size
                    rate = total_roms / total_time if total_time > 0 else 0
                    
                    print(f"Batch {batches_tested}: {completed}/{batch_size} completed, "
                          f"{interesting} interesting, {ca_count} CA patterns")
                    print(f"Totals: {total_roms:,} tested, {completed_total} completed, "
                          f"{interesting_total} interesting, {ca_total} CA patterns ({rate:.0f} ROMs/sec)")
                    
                    if ca_total > 0:
                        print(f"CA discovery rate: 1 in {total_roms // ca_total:,} ROMs")
                    print("=" * 50)
                    
            except KeyboardInterrupt:
                print(f"\nStopped after {batches_tested} batches")
                if ca_total > 0:
                    print(f"Found {ca_total} CA patterns in {batches_tested * batch_size:,} ROMs")
                
        elif choice == "3":
            rom_dir = input("ROM directory path: ").strip()
            if os.path.exists(rom_dir):
                rom_files = load_rom_files(rom_dir)
                if rom_files:
                    rom_data_list = [rom_data for _, rom_data in rom_files]
                    results, ca_roms = test_rom_batch(rom_data_list, cycles=args.cycles, ca_detector=ca_detector)
                    print_summary_stats(results)
                    
                    if ca_roms:
                        save_ca_roms_only(ca_roms, "output/ca_roms")
                else:
                    print("No ROM files found")
            else:
                print("Directory not found")
        
        elif choice == "4":
            # Analyze existing interesting ROMs for CA patterns
            interesting_dir = input("Interesting ROMs directory (default: output/interesting_roms): ").strip()
            if not interesting_dir:
                interesting_dir = "output/interesting_roms"
            
            if os.path.exists(interesting_dir):
                rom_files = load_rom_files(interesting_dir)
                if rom_files:
                    print(f"Analyzing {len(rom_files)} existing ROMs for CA patterns...")
                    rom_data_list = [rom_data for _, rom_data in rom_files]
                    results, ca_roms = test_rom_batch(rom_data_list, cycles=args.cycles, ca_detector=ca_detector)
                    
                    ca_count = len(ca_roms)
                    print(f"Found {ca_count} ROMs with CA patterns out of {len(rom_files)} analyzed")
                    
                    if ca_roms:
                        save_ca_roms_only(ca_roms, "output/ca_roms_from_existing")
                        
                        # Print top CA candidates
                        sorted_ca = sorted(ca_roms, key=lambda x: x[1].ca_likelihood, reverse=True)
                        print(f"\nTop CA candidates:")
                        for i, (rom_data, ca_analysis) in enumerate(sorted_ca[:5]):
                            print(f"{i+1}. Likelihood: {ca_analysis.ca_likelihood:.1f}% - {ca_analysis.classification}")
                else:
                    print("No ROM files found")
            else:
                print("Directory not found")
        
        else:
            print("Invalid choice")
            return 1
    
    print("\nTesting completed!")
    return 0


if __name__ == "__main__":
    exit(main())