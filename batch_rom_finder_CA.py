#!/usr/bin/env python3
"""
High-Selectivity CA ROM Detector
Only saves ROMs with CA likelihood ≥ 80%
Much stricter criteria to avoid false positives
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


class HighSelectivityCADetector:
    """Ultra-selective memory-based CA detector"""
    
    def __init__(self):
        # Minimum thresholds for memory-based CA detection
        self.min_ca_likelihood = 80.0
        self.min_execution_percentage = 70.0  # Hot loop must dominate execution
        self.min_instructions = 8000  # Must run for substantial time
        self.min_memory_operations = 10  # Must actively read/write memory
        
    def analyze_rom_for_ca(self, rom_data: bytes, max_cycles: int = 50000) -> Optional[CAAnalysis]:
        """Memory-focused CA analysis - looks for CA patterns in memory operations"""
        if Chip8Emulator is None:
            return None
        
        try:
            # Create emulator and load ROM
            emulator = Chip8Emulator()
            emulator.load_rom(rom_data)
            
            # Track execution patterns
            pc_frequency = defaultdict(int)
            total_cycles = 0
            memory_operations = 0
            
            # Run ROM and track memory activity
            for cycle in range(max_cycles):
                if emulator.crashed or emulator.halt:
                    break
                
                pc = emulator.program_counter
                pc_frequency[pc] += 1
                
                # Track memory operations
                if pc < 4094:
                    instruction = (int(emulator.memory[pc]) << 8) | int(emulator.memory[pc + 1])
                    if (instruction & 0xF0FF) in [0xF055, 0xF065]:  # Memory read/write
                        memory_operations += 1
                
                if not emulator.step():
                    break
                
                total_cycles += 1
                
                # Early exit for non-memory-intensive programs
                if cycle > 15000 and memory_operations < 5:
                    break
            
            # Pre-filter: Must meet basic memory-CA criteria
            if total_cycles < self.min_instructions:
                return None
            
            if memory_operations < self.min_memory_operations:
                return None
            
            # Find hot execution region
            if not pc_frequency:
                return None
            
            sorted_pcs = sorted(pc_frequency.items(), key=lambda x: x[1], reverse=True)
            total_executions = sum(pc_frequency.values())
            
            # Find the most concentrated execution region
            hot_pc_start = sorted_pcs[0][0]
            hot_pc_end = hot_pc_start
            hot_execution_count = sorted_pcs[0][1]
            
            # Build tight execution cluster
            for pc, count in sorted_pcs[1:20]:  # Check top 20 most frequent
                if abs(pc - hot_pc_start) <= 20:  # Reasonable cluster
                    hot_pc_start = min(hot_pc_start, pc)
                    hot_pc_end = max(hot_pc_end, pc + 2)
                    hot_execution_count += count
            
            execution_percentage = hot_execution_count / total_executions * 100
            
            # Execution concentration requirement
            if execution_percentage < self.min_execution_percentage:
                return None
            
            # Analyze for memory-based CA patterns
            ca_score = 0.0
            evidence = []
            
            # Check memory-based CA patterns
            memory_sequential = self._check_memory_grid_access(emulator.memory, hot_pc_start, hot_pc_end)
            state_evolution = self._check_memory_state_evolution(emulator.memory, hot_pc_start, hot_pc_end)
            neighbor_checking = self._check_memory_neighbor_patterns(emulator.memory, hot_pc_start, hot_pc_end)
            ca_rules = self._check_ca_rule_patterns(emulator.memory, hot_pc_start, hot_pc_end)
            
            # Score memory-based CA patterns
            if memory_sequential:
                ca_score += 40
                evidence.append("Sequential memory grid access pattern detected")
            
            if state_evolution:
                ca_score += 35
                evidence.append("Memory-based state evolution cycle confirmed")
            
            if neighbor_checking:
                ca_score += 35
                evidence.append("Multi-directional memory neighbor access detected")
            
            if ca_rules:
                ca_score += 25
                evidence.append("CA-like computational rules in memory operations")
            
            # Additional memory-focused scoring
            memory_complexity = self._analyze_memory_access_complexity(emulator.memory, hot_pc_start, hot_pc_end)
            if memory_complexity >= 4:
                ca_score += 15
                evidence.append("Complex memory access patterns suggesting 2D grid")
            
            # Check for memory-based CA instruction sequences
            memory_ca_patterns = self._check_memory_ca_sequences(emulator.memory, hot_pc_start, hot_pc_end)
            if memory_ca_patterns:
                ca_score += 20
                evidence.append("Classic memory-CA instruction sequences detected")
            
            # Ultra-strict threshold for memory-based CAs
            if ca_score < self.min_ca_likelihood:
                return None
            
            # Additional validation: Must be memory-focused, not display-focused
            if not self._is_memory_focused_ca(emulator.memory, hot_pc_start, hot_pc_end):
                return None
            
            # Classify the memory-based CA pattern
            classification = self._classify_memory_ca_pattern(emulator.memory, hot_pc_start, hot_pc_end, 
                                                            memory_sequential, state_evolution, neighbor_checking)
            
            rule_complexity = self._assess_memory_rule_complexity(emulator.memory, hot_pc_start, hot_pc_end)
            
            # Note: display_patterns is not relevant for memory-based CAs
            return CAAnalysis(
                ca_likelihood=min(ca_score, 100.0),
                memory_sequential=memory_sequential,
                state_evolution=state_evolution,
                display_patterns=False,  # Not relevant for memory CAs
                neighbor_checking=neighbor_checking,
                rule_complexity=rule_complexity,
                classification=classification,
                evidence=evidence,
                hot_loop_pc_range=(hot_pc_start, hot_pc_end),
                execution_percentage=execution_percentage
            )
            
        except Exception as e:
            return None
    
    def _check_memory_grid_access(self, memory: np.ndarray, start: int, end: int) -> bool:
        """Check for memory access patterns suggesting 2D grid operations"""
        add_i_operations = []  # Track different index manipulations
        memory_reads = 0
        memory_writes = 0
        index_loads = 0
        
        for addr in range(start, min(end + 1, len(memory) - 1), 2):
            instruction = (int(memory[addr]) << 8) | int(memory[addr + 1])
            
            # Track index register operations
            if (instruction & 0xF0FF) == 0xF01E:  # ADD I, Vx
                vx = (instruction & 0x0F00) >> 8
                add_i_operations.append(vx)
            
            # Memory operations
            if (instruction & 0xF0FF) == 0xF065:  # LD Vx, [I]
                memory_reads += 1
            elif (instruction & 0xF0FF) == 0xF055:  # LD [I], Vx
                memory_writes += 1
            
            # Index loading
            if (instruction & 0xF000) == 0xA000:  # LD I, addr
                index_loads += 1
        
        # Must have grid-like access: multiple index manipulations + memory ops
        unique_index_ops = len(set(add_i_operations))
        return (unique_index_ops >= 2 and memory_reads >= 3 and 
                memory_writes >= 2 and index_loads >= 1)
    
    def _check_memory_state_evolution(self, memory: np.ndarray, start: int, end: int) -> bool:
        """Check for memory-based state evolution (read current state, compute new state, write back)"""
        memory_reads = 0
        memory_writes = 0
        computational_ops = 0
        has_conditionals = False
        
        for addr in range(start, min(end + 1, len(memory) - 1), 2):
            instruction = (int(memory[addr]) << 8) | int(memory[addr + 1])
            opcode = (instruction & 0xF000) >> 12
            
            # Memory state access
            if (instruction & 0xF0FF) == 0xF065:  # LD Vx, [I] - read state
                memory_reads += 1
            elif (instruction & 0xF0FF) == 0xF055:  # LD [I], Vx - write state
                memory_writes += 1
            
            # Computational operations (CA rules)
            elif opcode == 0x8:  # Register operations
                subop = instruction & 0x000F
                if subop in [0x1, 0x2, 0x3, 0x4, 0x5]:  # OR, AND, XOR, ADD, SUB
                    computational_ops += 1
            
            # Conditional operations (state-dependent rules)
            elif opcode in [0x3, 0x4, 0x5, 0x9]:  # Skip instructions
                has_conditionals = True
        
        # Must have read-compute-write cycle with substantial computation
        return (memory_reads >= 2 and memory_writes >= 2 and 
                computational_ops >= 3)
    
    def _check_memory_neighbor_patterns(self, memory: np.ndarray, start: int, end: int) -> bool:
        """Check for neighbor cell access patterns in memory"""
        index_modifications = []
        memory_accesses = 0
        different_offsets = set()
        
        for addr in range(start, min(end + 1, len(memory) - 1), 2):
            instruction = (int(memory[addr]) << 8) | int(memory[addr + 1])
            
            # Index register modifications (moving through grid)
            if (instruction & 0xF0FF) == 0xF01E:  # ADD I, Vx
                vx = (instruction & 0x0F00) >> 8
                index_modifications.append(vx)
            
            # Register loading with immediate values (offsets)
            elif (instruction & 0xF000) == 0x6000:  # LD Vx, byte
                offset = instruction & 0x00FF
                if offset <= 64:  # Reasonable grid offset
                    different_offsets.add(offset)
            
            # Memory access
            elif (instruction & 0xF0FF) in [0xF055, 0xF065]:
                memory_accesses += 1
        
        # Must access multiple different memory locations (neighbors)
        unique_index_mods = len(set(index_modifications))
        return (unique_index_mods >= 2 and memory_accesses >= 4 and 
                len(different_offsets) >= 2)
    
    def _check_ca_rule_patterns(self, memory: np.ndarray, start: int, end: int) -> bool:
        """Check for CA-like computational rules"""
        xor_operations = 0
        and_operations = 0
        or_operations = 0
        conditional_operations = 0
        
        for addr in range(start, min(end + 1, len(memory) - 1), 2):
            instruction = (int(memory[addr]) << 8) | int(memory[addr + 1])
            opcode = (instruction & 0xF000) >> 12
            
            if opcode == 0x8:  # Register operations
                subop = instruction & 0x000F
                if subop == 0x3:  # XOR - classic CA operation
                    xor_operations += 1
                elif subop == 0x2:  # AND
                    and_operations += 1
                elif subop == 0x1:  # OR
                    or_operations += 1
            
            # Conditional operations (state-dependent rules)
            elif opcode in [0x3, 0x4, 0x5, 0x9]:
                conditional_operations += 1
        
        # CA-like rules: must have logical operations and state-dependent behavior
        total_logic_ops = xor_operations + and_operations + or_operations
        return (total_logic_ops >= 2 and 
                (xor_operations >= 1 or conditional_operations >= 2))
    
    def _analyze_memory_access_complexity(self, memory: np.ndarray, start: int, end: int) -> int:
        """Analyze complexity of memory access patterns"""
        complexity_score = 0
        unique_registers_used = set()
        index_operations = 0
        
        for addr in range(start, min(end + 1, len(memory) - 1), 2):
            instruction = (int(memory[addr]) << 8) | int(memory[addr + 1])
            opcode = (instruction & 0xF000) >> 12
            
            # Track register usage
            if opcode in [0x6, 0x7, 0x8, 0xF]:
                x = (instruction & 0x0F00) >> 8
                unique_registers_used.add(x)
            
            # Index register operations
            if (instruction & 0xF0FF) in [0xF01E, 0xA000]:  # ADD I, Vx or LD I, addr
                index_operations += 1
            
            # Memory operations
            if (instruction & 0xF0FF) in [0xF055, 0xF065]:
                complexity_score += 2
            
            # Computational operations
            if opcode == 0x8:
                complexity_score += 1
        
        # Bonus for using multiple registers (suggests 2D coordinates)
        if len(unique_registers_used) >= 3:
            complexity_score += 2
        
        return complexity_score
    
    def _check_memory_ca_sequences(self, memory: np.ndarray, start: int, end: int) -> bool:
        """Check for classic memory-based CA instruction sequences"""
        instructions = []
        for addr in range(start, min(end + 1, len(memory) - 1), 2):
            instruction = (int(memory[addr]) << 8) | int(memory[addr + 1])
            instructions.append(instruction)
        
        # Look for patterns like: LD Vx, [I] -> XOR/AND/OR -> LD [I], Vx
        for i in range(len(instructions) - 2):
            instr1 = instructions[i]
            instr2 = instructions[i + 1]
            instr3 = instructions[i + 2]
            
            # Memory read -> Logic operation -> Memory write
            if ((instr1 & 0xF0FF) == 0xF065 and  # LD Vx, [I]
                (instr2 & 0xF000) == 0x8000 and  # Register operation
                (instr3 & 0xF0FF) == 0xF055):     # LD [I], Vx
                return True
        
        return False
    
    def _is_memory_focused_ca(self, memory: np.ndarray, start: int, end: int) -> bool:
        """Verify this is a memory-focused CA, not display-focused"""
        memory_operations = 0
        display_operations = 0
        
        for addr in range(start, min(end + 1, len(memory) - 1), 2):
            instruction = (int(memory[addr]) << 8) | int(memory[addr + 1])
            opcode = (instruction & 0xF000) >> 12
            
            if (instruction & 0xF0FF) in [0xF055, 0xF065]:  # Memory ops
                memory_operations += 1
            elif opcode == 0xD:  # Display operations
                display_operations += 1
        
        # Must be memory-focused: more memory ops than display ops
        return memory_operations >= display_operations * 2
    
    def _classify_memory_ca_pattern(self, memory: np.ndarray, start: int, end: int,
                                  sequential: bool, evolution: bool, neighbor: bool) -> str:
        """Classify memory-based CA patterns"""
        
        has_xor = any((int(memory[addr]) << 8 | int(memory[addr + 1])) & 0xF00F == 0x8003
                     for addr in range(start, min(end + 1, len(memory) - 1), 2))
        
        has_conditionals = any((int(memory[addr]) << 8 | int(memory[addr + 1])) & 0xF000 in [0x3000, 0x4000, 0x5000, 0x9000]
                              for addr in range(start, min(end + 1, len(memory) - 1), 2))
        
        if has_xor and evolution and neighbor:
            return "Memory-based XOR cellular automaton"
        elif sequential and evolution and neighbor:
            return "Memory-grid cellular automaton"
        elif has_conditionals and evolution:
            return "Conditional memory-state automaton" 
        elif sequential and evolution:
            return "Linear memory cellular automaton"
        else:
            return "Memory-based computational automaton"
    
    def _assess_memory_rule_complexity(self, memory: np.ndarray, start: int, end: int) -> str:
        """Assess complexity of memory-based CA rules"""
        logical_ops = 0
        arithmetic_ops = 0
        conditional_ops = 0
        
        for addr in range(start, min(end + 1, len(memory) - 1), 2):
            instruction = (int(memory[addr]) << 8) | int(memory[addr + 1])
            opcode = (instruction & 0xF000) >> 12
            
            if opcode == 0x8:
                subop = instruction & 0x000F
                if subop in [0x1, 0x2, 0x3]:  # OR, AND, XOR
                    logical_ops += 1
                elif subop in [0x4, 0x5, 0x7]:  # ADD, SUB, SUBN
                    arithmetic_ops += 1
            elif opcode in [0x3, 0x4, 0x5, 0x9]:  # Conditionals
                conditional_ops += 1
        
        total_complexity = logical_ops + arithmetic_ops + conditional_ops
        
        if total_complexity >= 6:
            return "complex"
        elif total_complexity >= 3:
            return "moderate"
        else:
            return "simple"
    
    def _is_likely_false_positive(self, memory: np.ndarray, start: int, end: int, stats: dict) -> bool:
        """Check for common false positive patterns"""
        
        # Too many jumps suggests chaotic behavior, not CA
        jump_count = 0
        total_instructions = 0
        
        for addr in range(start, min(end + 1, len(memory) - 1), 2):
            instruction = (int(memory[addr]) << 8) | int(memory[addr + 1])
            opcode = (instruction & 0xF000) >> 12
            total_instructions += 1
            
            if opcode in [0x1, 0x2]:  # JP, CALL
                jump_count += 1
        
        # If more than 30% jumps, likely not a clean CA
        if total_instructions > 0 and jump_count / total_instructions > 0.3:
            return True
        
        # Check for excessive randomness (RND instructions)
        rnd_count = sum(1 for addr in range(start, min(end + 1, len(memory) - 1), 2)
                       if (int(memory[addr]) << 8 | int(memory[addr + 1])) & 0xF000 == 0xC000)
        
        if rnd_count > 2:  # Too much randomness for pure CA
            return True
        
        return False
    
    def _classify_strict_ca_pattern(self, memory: np.ndarray, start: int, end: int,
                                  sequential: bool, evolution: bool, neighbor: bool) -> str:
        """Classify high-confidence CA patterns"""
        
        has_xor = any((int(memory[addr]) << 8 | int(memory[addr + 1])) & 0xF00F == 0x8003
                     for addr in range(start, min(end + 1, len(memory) - 1), 2))
        
        has_display = any((int(memory[addr]) << 8 | int(memory[addr + 1])) & 0xF000 == 0xD000
                         for addr in range(start, min(end + 1, len(memory) - 1), 2))
        
        if has_xor and has_display and evolution and neighbor:
            return "High-confidence XOR-based 2D cellular automaton"
        elif sequential and evolution and neighbor:
            return "High-confidence grid-based cellular automaton"
        elif has_xor and evolution:
            return "High-confidence XOR-rule cellular automaton"
        elif sequential and evolution:
            return "High-confidence linear cellular automaton"
        else:
            return "High-confidence computational automaton"
    
    def _assess_strict_rule_complexity(self, memory: np.ndarray, start: int, end: int) -> str:
        """Assess CA rule complexity with strict criteria"""
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
        
        if total_ops >= 4:
            return "complex"
        elif total_ops >= 2:
            return "moderate"
        else:
            return "simple"


def load_rom_files(rom_dir: str) -> List[Tuple[str, bytes]]:
    """Load all ROM files from a directory"""
    rom_files = []
    
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
                  ca_detector: Optional[HighSelectivityCADetector] = None) -> Tuple[List[Dict], List[Tuple[bytes, CAAnalysis]]]:
    """Test batch with ultra-selective CA detection"""
    print(f"Testing batch of {len(rom_data_list)} ROMs with high-selectivity CA detection...")
    
    emulator = ParallelChip8Emulator(len(rom_data_list))
    emulator.load_roms(rom_data_list)
    
    start_time = time.time()
    emulator.run(cycles=cycles)
    execution_time = time.time() - start_time
    
    print(f"Batch execution completed in {execution_time:.2f} seconds")
    
    results = []
    displays = emulator.get_displays()
    ca_roms = []
    
    for i in range(len(rom_data_list)):
        final_pc = int(emulator.program_counter[i])
        crashed = bool(emulator.crashed[i])
        halted = bool(emulator.halted[i])
        waiting_for_key = bool(emulator.waiting_for_key[i])
        
        display = displays[i]
        pixels_set = int(np.sum(display > 0))
        total_pixels = display.shape[0] * display.shape[1]
        pixel_density = pixels_set / total_pixels
        
        has_output = pixels_set > 0
        has_structure = pixel_density > 0.05 and pixel_density < 0.5
        
        instructions = int(emulator.stats['instructions_executed'][i])
        display_writes = int(emulator.stats['display_writes'][i])
        pixels_drawn = int(emulator.stats['pixels_drawn'][i])
        
        # More selective criteria for memory-based CA analysis
        potentially_ca = (not crashed and not waiting_for_key and 
                         instructions > 8000 and 
                         # Focus on memory activity, not display
                         int(emulator.stats.get('memory_reads', 0)) + int(emulator.stats.get('memory_writes', 0)) > 10)
        
        result = {
            'execution_time': execution_time / len(rom_data_list),
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
            'ca_analysis': None
        }
        
        # Run high-selectivity memory-based CA detection
        if potentially_ca and ca_detector is not None:
            ca_analysis = ca_detector.analyze_rom_for_ca(rom_data_list[i], max_cycles=cycles//5)
            if ca_analysis is not None:  # Only saves 80%+ likelihood
                result['ca_analysis'] = ca_analysis
                ca_roms.append((rom_data_list[i], ca_analysis))
                print(f"  🧬 MEMORY-BASED CA! Likelihood: {ca_analysis.ca_likelihood:.0f}% - {ca_analysis.classification}")
        
        results.append(result)
    
    return results, ca_roms


def save_high_confidence_ca_roms(ca_roms: List[Tuple[bytes, CAAnalysis]], output_dir: str = "memory_based_ca_roms"):
    """Save only ultra-high confidence memory-based CA ROMs"""
    os.makedirs(output_dir, exist_ok=True)
    
    for i, (rom_data, ca_analysis) in enumerate(ca_roms):
        likelihood = int(ca_analysis.ca_likelihood)
        classification = ca_analysis.classification.replace(' ', '_').replace('-', '_')
        
        filename = f"MemoryCA{likelihood:02d}_{classification}_{i:04d}"
        
        # Save ROM
        rom_path = os.path.join(output_dir, f"{filename}.ch8")
        with open(rom_path, 'wb') as f:
            f.write(rom_data)
        
        # Save detailed analysis
        analysis_path = os.path.join(output_dir, f"{filename}_ANALYSIS.txt")
        with open(analysis_path, 'w') as f:
            f.write(f"MEMORY-BASED CELLULAR AUTOMATON ANALYSIS\n")
            f.write(f"=" * 50 + "\n\n")
            f.write(f"CA Likelihood: {ca_analysis.ca_likelihood:.1f}%\n")
            f.write(f"Classification: {ca_analysis.classification}\n")
            f.write(f"Rule Complexity: {ca_analysis.rule_complexity}\n")
            f.write(f"Hot Loop: 0x{ca_analysis.hot_loop_pc_range[0]:03X}-0x{ca_analysis.hot_loop_pc_range[1]:03X}\n")
            f.write(f"Execution Concentration: {ca_analysis.execution_percentage:.1f}%\n\n")
            f.write(f"Memory-Based CA Features:\n")
            f.write(f"✓ Memory Grid Access: {'YES' if ca_analysis.memory_sequential else 'NO'}\n")
            f.write(f"✓ State Evolution: {'YES' if ca_analysis.state_evolution else 'NO'}\n")
            f.write(f"✓ Neighbor Checking: {'YES' if ca_analysis.neighbor_checking else 'NO'}\n")
            f.write(f"✓ Memory-Focused: YES (not display-based)\n\n")
            f.write(f"Evidence:\n")
            for evidence in ca_analysis.evidence:
                f.write(f"• {evidence}\n")
    
    print(f"📁 Saved {len(ca_roms)} MEMORY-BASED CA ROMs to {output_dir}/")
    return len(ca_roms)


def generate_and_test_random_batch(generator, num_roms: int = 10000, 
                                 test_cycles: int = 1000000,
                                 ca_detector: Optional[HighSelectivityCADetector] = None) -> Tuple[int, int, int]:
    """Generate and test with ultra-selective CA detection"""
    print(f"Generating and testing {num_roms:,} ROMs with 80%+ CA threshold...")
    
    roms = generator.generate_batch(num_roms)
    rom_data_list = [cp.asnumpy(rom).tobytes() for rom in roms]
    
    results, ca_roms = test_rom_batch(rom_data_list, cycles=test_cycles, ca_detector=ca_detector)
    
    interesting_count = sum(1 for r in results if r['interesting'])
    completed_count = sum(1 for r in results if r['completed_normally'])
    ca_count = len(ca_roms)
    
    print(f"Results: {completed_count} completed, {interesting_count} interesting, {ca_count} HIGH-CONFIDENCE CAs")
    
    if ca_count > 0:
        save_high_confidence_ca_roms(ca_roms, "output/memory_based_ca_roms")
    
    return completed_count, interesting_count, ca_count


def main():
    """Ultra-selective CA detection main"""
    parser = argparse.ArgumentParser(description="High-Selectivity CA ROM Detector (80%+ only)")
    parser.add_argument("--generate", type=int, help="Generate N random ROMs")
    parser.add_argument("--continuous", action="store_true", help="Run continuously")
    parser.add_argument("--cycles", type=int, default=1000000, help="Cycles per ROM")
    parser.add_argument("--batch-size", type=int, default=20000, help="Batch size")
    
    args = parser.parse_args()
    
    print("Memory-Based CHIP-8 CA Detector (80%+ Confidence Only)")
    print("=" * 60)
    print("🧠 Focuses on memory-based cellular automata patterns")
    print("🎯 Only saves ROMs with CA likelihood ≥ 80%")
    print("📊 Ignores display output - pure memory CA detection")
    print()
    
    ca_detector = HighSelectivityCADetector()
    
    if args.continuous:
        if PureRandomChip8Generator is None:
            print("Error: Pure random ROM generator not available")
            return 1
        
        generator = PureRandomChip8Generator(rom_size=3584)
        completed_total = 0
        interesting_total = 0
        ca_total = 0
        batches_tested = 0
        start_time = time.time()
        
        try:
            while True:
                completed, interesting, ca_count = generate_and_test_random_batch(
                    generator, args.batch_size, args.cycles, ca_detector
                )
                
                completed_total += completed
                interesting_total += interesting
                ca_total += ca_count
                batches_tested += 1
                
                total_roms = batches_tested * args.batch_size
                total_time = time.time() - start_time
                rate = total_roms / total_time if total_time > 0 else 0
                
                print(f"Batch {batches_tested}: {ca_count} HIGH-CONFIDENCE CAs found")
                print(f"Totals: {total_roms:,} tested, {ca_total} HIGH-CONFIDENCE CAs ({rate:.0f} ROMs/sec)")
                
                if ca_total > 0:
                    print(f"🎯 CA discovery rate: 1 in {total_roms // ca_total:,} ROMs")
                print("=" * 60)
                
        except KeyboardInterrupt:
            print(f"\nStopped after {batches_tested} batches")
            if ca_total > 0:
                print(f"🧬 Found {ca_total} HIGH-CONFIDENCE CAs in {batches_tested * args.batch_size:,} ROMs")
    
    elif args.generate:
        if PureRandomChip8Generator is None:
            print("Error: Pure random ROM generator not available")
            return 1
        
        generator = PureRandomChip8Generator(rom_size=3584)
        completed_total = 0
        interesting_total = 0
        ca_total = 0
        
        total_tested = 0
        while total_tested < args.generate:
            batch_size = min(args.batch_size, args.generate - total_tested)
            completed, interesting, ca_count = generate_and_test_random_batch(
                generator, batch_size, args.cycles, ca_detector
            )
            
            completed_total += completed
            interesting_total += interesting
            ca_total += ca_count
            total_tested += batch_size
            
            print(f"Progress: {total_tested}/{args.generate} - {ca_total} HIGH-CONFIDENCE CAs found")
        
        print(f"\n🎯 Final: {ca_total} HIGH-CONFIDENCE CAs in {total_tested:,} ROMs")
    
    else:
        print("Use --continuous or --generate to start CA hunting")
    
    return 0


if __name__ == "__main__":
    exit(main())