"""
CHIP-8 ROM Decompiler + Cellular Automata Detector
For Babelscope computational archaeology project

Analyzes random CHIP-8 ROMs to find accidental cellular automata implementations
Focuses on hot execution regions and CA-like patterns in tight loops
"""

import numpy as np
import os
import glob
import time
from typing import Dict, List, Tuple, Optional, Set
from collections import defaultdict, Counter
from dataclasses import dataclass
import json

# Import the CHIP-8 emulator
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'emulators'))
from chip8 import Chip8Emulator

@dataclass
class HotLoop:
    """Represents a frequently executed code region"""
    start_addr: int
    end_addr: int
    instructions: List[Tuple[int, int, str]]  # (addr, opcode, disassembly)
    execution_count: int
    execution_percentage: float
    memory_accesses: List[Tuple[str, int]]  # (type, address)
    display_writes: int

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

class CHIP8Disassembler:
    """Smart disassembler focused on hot execution regions"""
    
    def __init__(self):
        self.instruction_map = {
            0x0: self._decode_0xxx,
            0x1: lambda i, a: f"JP 0x{i & 0x0FFF:03X}",
            0x2: lambda i, a: f"CALL 0x{i & 0x0FFF:03X}",
            0x3: lambda i, a: f"SE V{(i & 0x0F00) >> 8:X}, 0x{i & 0x00FF:02X}",
            0x4: lambda i, a: f"SNE V{(i & 0x0F00) >> 8:X}, 0x{i & 0x00FF:02X}",
            0x5: lambda i, a: f"SE V{(i & 0x0F00) >> 8:X}, V{(i & 0x00F0) >> 4:X}",
            0x6: lambda i, a: f"LD V{(i & 0x0F00) >> 8:X}, 0x{i & 0x00FF:02X}",
            0x7: lambda i, a: f"ADD V{(i & 0x0F00) >> 8:X}, 0x{i & 0x00FF:02X}",
            0x8: self._decode_8xxx,
            0x9: lambda i, a: f"SNE V{(i & 0x0F00) >> 8:X}, V{(i & 0x00F0) >> 4:X}",
            0xA: lambda i, a: f"LD I, 0x{i & 0x0FFF:03X}",
            0xB: lambda i, a: f"JP V0, 0x{i & 0x0FFF:03X}",
            0xC: lambda i, a: f"RND V{(i & 0x0F00) >> 8:X}, 0x{i & 0x00FF:02X}",
            0xD: lambda i, a: f"DRW V{(i & 0x0F00) >> 8:X}, V{(i & 0x00F0) >> 4:X}, {i & 0x000F}",
            0xE: self._decode_Exxx,
            0xF: self._decode_Fxxx,
        }
    
    def _decode_0xxx(self, instruction: int, addr: int) -> str:
        if instruction == 0x00E0:
            return "CLS"
        elif instruction == 0x00EE:
            return "RET"
        else:
            return f"SYS 0x{instruction & 0x0FFF:03X}"
    
    def _decode_8xxx(self, instruction: int, addr: int) -> str:
        x = (instruction & 0x0F00) >> 8
        y = (instruction & 0x00F0) >> 4
        n = instruction & 0x000F
        
        ops = {
            0x0: f"LD V{x:X}, V{y:X}",
            0x1: f"OR V{x:X}, V{y:X}",
            0x2: f"AND V{x:X}, V{y:X}",
            0x3: f"XOR V{x:X}, V{y:X}",
            0x4: f"ADD V{x:X}, V{y:X}",
            0x5: f"SUB V{x:X}, V{y:X}",
            0x6: f"SHR V{x:X}",
            0x7: f"SUBN V{x:X}, V{y:X}",
            0xE: f"SHL V{x:X}",
        }
        return ops.get(n, f"Unknown 8xy{n:X}")
    
    def _decode_Exxx(self, instruction: int, addr: int) -> str:
        x = (instruction & 0x0F00) >> 8
        kk = instruction & 0x00FF
        
        if kk == 0x9E:
            return f"SKP V{x:X}"
        elif kk == 0xA1:
            return f"SKNP V{x:X}"
        else:
            return f"Unknown Ex{kk:02X}"
    
    def _decode_Fxxx(self, instruction: int, addr: int) -> str:
        x = (instruction & 0x0F00) >> 8
        kk = instruction & 0x00FF
        
        ops = {
            0x07: f"LD V{x:X}, DT",
            0x0A: f"LD V{x:X}, K",
            0x15: f"LD DT, V{x:X}",
            0x18: f"LD ST, V{x:X}",
            0x1E: f"ADD I, V{x:X}",
            0x29: f"LD F, V{x:X}",
            0x33: f"LD B, V{x:X}",
            0x55: f"LD [I], V{x:X}",
            0x65: f"LD V{x:X}, [I]",
        }
        return ops.get(kk, f"Unknown Fx{kk:02X}")
    
    def disassemble_instruction(self, instruction: int, addr: int) -> str:
        """Disassemble a single instruction"""
        opcode = (instruction & 0xF000) >> 12
        return self.instruction_map.get(opcode, lambda i, a: f"Unknown 0x{i:04X}")(instruction, addr)
    
    def disassemble_region(self, memory: np.ndarray, start: int, end: int) -> List[Tuple[int, int, str]]:
        """Disassemble a memory region"""
        instructions = []
        addr = start
        
        while addr < end and addr < len(memory) - 1:
            instruction = (int(memory[addr]) << 8) | int(memory[addr + 1])
            disassembly = self.disassemble_instruction(instruction, addr)
            instructions.append((addr, instruction, disassembly))
            addr += 2
        
        return instructions

class ExecutionTracker:
    """Tracks execution patterns during ROM analysis"""
    
    def __init__(self):
        self.pc_frequency = defaultdict(int)
        self.memory_reads = []
        self.memory_writes = []
        self.display_changes = []
        self.total_cycles = 0
    
    def track_execution(self, emulator: Chip8Emulator, max_cycles: int = 100000):
        """Track execution patterns of a ROM"""
        self.pc_frequency.clear()
        self.memory_reads.clear()
        self.memory_writes.clear()
        self.display_changes.clear()
        self.total_cycles = 0
        
        prev_display = emulator.get_display().copy()
        
        for cycle in range(max_cycles):
            if emulator.crashed or emulator.halt:
                break
            
            # Track PC frequency
            pc = emulator.program_counter
            self.pc_frequency[pc] += 1
            
            # Store memory state before step
            prev_memory = emulator.memory.copy()
            
            # Execute one step
            if not emulator.step():
                break
            
            # Check for memory changes (simplified)
            # In a full implementation, you'd instrument the emulator
            # to track exact read/write addresses
            
            # Check for display changes
            curr_display = emulator.get_display()
            if not np.array_equal(curr_display, prev_display):
                self.display_changes.append(cycle)
                prev_display = curr_display.copy()
            
            self.total_cycles += 1
            
            # Break if we're in an infinite loop with no display changes
            if cycle > 10000 and len(self.display_changes) == 0:
                break
    
    def find_hot_loops(self, threshold: float = 0.8) -> List[HotLoop]:
        """Find frequently executed code regions"""
        if not self.pc_frequency:
            return []
        
        # Find addresses that account for most execution time
        sorted_pcs = sorted(self.pc_frequency.items(), key=lambda x: x[1], reverse=True)
        total_executions = sum(self.pc_frequency.values())
        
        hot_loops = []
        covered_execution = 0
        
        # Group consecutive addresses into loops
        current_region = []
        current_count = 0
        
        for pc, count in sorted_pcs:
            if covered_execution / total_executions >= threshold:
                break
            
            # Check if this PC is adjacent to current region
            if current_region and (pc < min(current_region) - 10 or pc > max(current_region) + 10):
                # Save current region if significant
                if current_count > total_executions * 0.05:  # At least 5% execution time
                    start_addr = min(current_region)
                    end_addr = max(current_region) + 2
                    hot_loops.append(HotLoop(
                        start_addr=start_addr,
                        end_addr=end_addr,
                        instructions=[],  # Will be filled by disassembler
                        execution_count=current_count,
                        execution_percentage=current_count / total_executions * 100,
                        memory_accesses=[],
                        display_writes=0
                    ))
                
                # Start new region
                current_region = [pc]
                current_count = count
            else:
                current_region.append(pc)
                current_count += count
            
            covered_execution += count
        
        # Don't forget the last region
        if current_region and current_count > total_executions * 0.05:
            start_addr = min(current_region)
            end_addr = max(current_region) + 2
            hot_loops.append(HotLoop(
                start_addr=start_addr,
                end_addr=end_addr,
                instructions=[],
                execution_count=current_count,
                execution_percentage=current_count / total_executions * 100,
                memory_accesses=[],
                display_writes=0
            ))
        
        return hot_loops

class CellularAutomataDetector:
    """Detects CA-like patterns in CHIP-8 programs"""
    
    def analyze_loop(self, loop: HotLoop, memory: np.ndarray) -> CAAnalysis:
        """Analyze a hot loop for CA-like behavior"""
        evidence = []
        ca_score = 0.0
        
        # Analyze instruction patterns
        memory_sequential = self._check_sequential_memory_access(loop.instructions)
        state_evolution = self._check_state_evolution(loop.instructions)
        display_patterns = self._check_display_patterns(loop)
        neighbor_checking = self._check_neighbor_access(loop.instructions)
        
        # Score CA likelihood
        if memory_sequential:
            ca_score += 25
            evidence.append("Sequential memory access detected")
        
        if state_evolution:
            ca_score += 30
            evidence.append("State evolution pattern found")
        
        if display_patterns:
            ca_score += 20
            evidence.append("Structured display updates")
        
        if neighbor_checking:
            ca_score += 25
            evidence.append("Neighbor-checking behavior")
        
        # Classify rule complexity
        rule_complexity = self._classify_rule_complexity(loop.instructions)
        classification = self._classify_ca_type(loop.instructions, memory_sequential, state_evolution)
        
        return CAAnalysis(
            ca_likelihood=min(ca_score, 100.0),
            memory_sequential=memory_sequential,
            state_evolution=state_evolution,
            display_patterns=display_patterns,
            neighbor_checking=neighbor_checking,
            rule_complexity=rule_complexity,
            classification=classification,
            evidence=evidence
        )
    
    def _check_sequential_memory_access(self, instructions: List[Tuple[int, int, str]]) -> bool:
        """Check for sequential memory access patterns"""
        memory_ops = []
        for addr, opcode, disasm in instructions:
            # Look for memory operations
            if any(op in disasm for op in ["LD [I]", "LD V", "[I]", "ADD I"]):
                memory_ops.append(disasm)
        
        # Look for patterns like: ADD I, Vx followed by memory operations
        return len(memory_ops) >= 2 and any("ADD I" in op for op in memory_ops)
    
    def _check_state_evolution(self, instructions: List[Tuple[int, int, str]]) -> bool:
        """Check for read-modify-write cycles"""
        has_read = False
        has_modify = False
        has_write = False
        
        for addr, opcode, disasm in instructions:
            if "LD V" in disasm and "[I]" in disasm:
                has_read = True
            if any(op in disasm for op in ["OR", "AND", "XOR", "ADD", "SUB"]):
                has_modify = True
            if "LD [I]" in disasm:
                has_write = True
        
        return has_read and has_modify and has_write
    
    def _check_display_patterns(self, loop: HotLoop) -> bool:
        """Check for structured display update patterns"""
        return loop.display_writes > 0 and loop.execution_percentage > 50
    
    def _check_neighbor_access(self, instructions: List[Tuple[int, int, str]]) -> bool:
        """Check for neighbor-checking patterns"""
        # Look for multiple memory reads with index manipulation
        index_ops = sum(1 for _, _, disasm in instructions if "ADD I" in disasm)
        memory_reads = sum(1 for _, _, disasm in instructions if "LD V" in disasm and "[I]" in disasm)
        
        return index_ops >= 2 and memory_reads >= 2
    
    def _classify_rule_complexity(self, instructions: List[Tuple[int, int, str]]) -> str:
        """Classify the complexity of the computational rule"""
        logical_ops = sum(1 for _, _, disasm in instructions 
                         if any(op in disasm for op in ["OR", "AND", "XOR"]))
        arithmetic_ops = sum(1 for _, _, disasm in instructions 
                           if any(op in disasm for op in ["ADD", "SUB"]))
        
        total_ops = logical_ops + arithmetic_ops
        
        if total_ops <= 2:
            return "simple"
        elif total_ops <= 5:
            return "moderate"
        else:
            return "complex"
    
    def _classify_ca_type(self, instructions: List[Tuple[int, int, str]], 
                         sequential: bool, evolution: bool) -> str:
        """Classify the type of cellular automaton"""
        if not (sequential or evolution):
            return "Non-CA pattern"
        
        # Look for specific patterns
        has_xor = any("XOR" in disasm for _, _, disasm in instructions)
        has_display = any("DRW" in disasm for _, _, disasm in instructions)
        has_increment = any("ADD V" in disasm for _, _, disasm in instructions)
        
        if has_xor and has_display:
            return "XOR-based visual automaton"
        elif has_increment and sequential:
            return "Linear growth automaton"
        elif evolution and has_display:
            return "State evolution automaton"
        elif sequential:
            return "Memory scanning pattern"
        else:
            return "Simple computational loop"

class ROMAnalyzer:
    """Main analyzer that coordinates all components"""
    
    def __init__(self):
        self.disassembler = CHIP8Disassembler()
        self.tracker = ExecutionTracker()
        self.ca_detector = CellularAutomataDetector()
    
    def analyze_rom(self, rom_path: str, max_cycles: int = 50000) -> Dict:
        """Analyze a single ROM file"""
        try:
            # Load and run ROM
            emulator = Chip8Emulator()
            emulator.load_rom(rom_path)
            
            # Track execution
            self.tracker.track_execution(emulator, max_cycles)
            
            # Find hot loops
            hot_loops = self.tracker.find_hot_loops()
            
            # Disassemble hot loops
            for loop in hot_loops:
                loop.instructions = self.disassembler.disassemble_region(
                    emulator.memory, loop.start_addr, loop.end_addr)
                loop.display_writes = emulator.stats.get('display_writes', 0)
            
            # Analyze for CA patterns
            ca_analyses = []
            for loop in hot_loops:
                ca_analysis = self.ca_detector.analyze_loop(loop, emulator.memory)
                ca_analyses.append(ca_analysis)
            
            # Find best CA candidate
            best_ca = max(ca_analyses, key=lambda x: x.ca_likelihood) if ca_analyses else None
            
            return {
                'rom_path': rom_path,
                'execution_stats': emulator.get_stats(),
                'hot_loops': hot_loops,
                'ca_analyses': ca_analyses,
                'best_ca_likelihood': best_ca.ca_likelihood if best_ca else 0,
                'crashed': emulator.crashed,
                'total_cycles': self.tracker.total_cycles,
                'display_changes': len(self.tracker.display_changes)
            }
            
        except Exception as e:
            return {
                'rom_path': rom_path,
                'error': str(e),
                'best_ca_likelihood': 0
            }
    
    def analyze_rom_directory(self, directory: str, output_file: str = "ca_analysis_results.json") -> List[Dict]:
        """Analyze all ROMs in a directory"""
        rom_files = glob.glob(os.path.join(directory, "*.ch8"))
        
        if not rom_files:
            print(f"No .ch8 files found in {directory}")
            return []
        
        print(f"Found {len(rom_files)} ROM files to analyze...")
        
        results = []
        start_time = time.time()
        
        for i, rom_path in enumerate(rom_files):
            if i % 100 == 0:
                elapsed = time.time() - start_time
                if i > 0:
                    eta = elapsed / i * (len(rom_files) - i)
                    print(f"Processed {i}/{len(rom_files)} ROMs ({i/len(rom_files)*100:.1f}%) - ETA: {eta/60:.1f} minutes")
                else:
                    print(f"Starting analysis of {len(rom_files)} ROMs...")
            
            result = self.analyze_rom(rom_path)
            results.append(result)
        
        # Sort by CA likelihood
        results.sort(key=lambda x: x.get('best_ca_likelihood', 0), reverse=True)
        
        # Save results
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        elapsed = time.time() - start_time
        print(f"\nAnalysis complete! Processed {len(rom_files)} ROMs in {elapsed/60:.1f} minutes")
        print(f"Results saved to {output_file}")
        
        return results
    
    def generate_report(self, rom_path: str, result: Dict) -> str:
        """Generate a detailed report for a single ROM"""
        if 'error' in result:
            return f"ROM: {os.path.basename(rom_path)}\nError: {result['error']}\n"
        
        report = []
        report.append(f"ROM: {os.path.basename(rom_path)}")
        
        if not result['hot_loops']:
            report.append("No significant hot loops found")
            return "\n".join(report) + "\n"
        
        # Find the most significant hot loop
        main_loop = max(result['hot_loops'], key=lambda x: x.execution_percentage)
        
        report.append(f"Hot Loop: 0x{main_loop.start_addr:03X}-0x{main_loop.end_addr:03X} "
                     f"({main_loop.execution_percentage:.1f}% execution time, "
                     f"{len(main_loop.instructions)} instructions)")
        report.append("")
        report.append("Assembly:")
        
        for addr, opcode, disasm in main_loop.instructions:
            report.append(f"0x{addr:03X}: {opcode:04X}  {disasm}")
        
        # Find corresponding CA analysis
        if result['ca_analyses']:
            ca_analysis = result['ca_analyses'][0]  # Assume first analysis corresponds to main loop
            
            report.append("")
            report.append("CA Analysis:")
            report.append(f"- Memory Access: {'Sequential ✓' if ca_analysis.memory_sequential else 'Random'}")
            report.append(f"- State Evolution: {'Yes ✓' if ca_analysis.state_evolution else 'No'}")
            report.append(f"- Display Pattern: {'Yes ✓' if ca_analysis.display_patterns else 'No'}")
            report.append(f"- Neighbor Checking: {'Yes ✓' if ca_analysis.neighbor_checking else 'No'}")
            report.append(f"- CA Likelihood: {ca_analysis.ca_likelihood:.0f}%")
            report.append(f"- Classification: {ca_analysis.classification}")
            
            if ca_analysis.evidence:
                report.append("- Evidence:")
                for evidence in ca_analysis.evidence:
                    report.append(f"  * {evidence}")
        
        return "\n".join(report) + "\n"

def main():
    """Main function to run the CA detector"""
    analyzer = ROMAnalyzer()
    
    # Analyze all ROMs in the interesting_roms directory
    rom_directory = "output/interesting_roms"
    
    # Check if directory exists
    if not os.path.exists(rom_directory):
        print(f"Directory {rom_directory} not found!")
        print("Please ensure you have the interesting ROMs in the correct location.")
        return
    
    print("CHIP-8 Cellular Automata Detector")
    print("=" * 50)
    
    # Run analysis
    results = analyzer.analyze_rom_directory(rom_directory)
    
    # Generate top 10 report
    print("\nTop 10 Most CA-like ROMs:")
    print("=" * 50)
    
    for i, result in enumerate(results[:10]):
        if result.get('best_ca_likelihood', 0) > 0:
            print(f"\n{i+1}. CA Likelihood: {result['best_ca_likelihood']:.0f}%")
            report = analyzer.generate_report(result['rom_path'], result)
            print(report)
    
    # Save detailed reports for top candidates
    with open("top_ca_candidates.txt", "w") as f:
        f.write("Top Cellular Automata Candidates\n")
        f.write("=" * 50 + "\n\n")
        
        for i, result in enumerate(results[:20]):
            if result.get('best_ca_likelihood', 0) > 20:  # Only high-likelihood candidates
                report = analyzer.generate_report(result['rom_path'], result)
                f.write(report + "\n" + "-" * 50 + "\n\n")
    
    print(f"\nDetailed reports for top candidates saved to: top_ca_candidates.txt")

if __name__ == "__main__":
    main()