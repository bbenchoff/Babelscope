#!/usr/bin/env python3
"""
CHIP-8 Sorting ROM Decompiler with Enhanced Authenticity Detection
Analyzes discovered sorting ROMs to determine genuine sorting vs. coincidental consecutive values

FIXED: Now properly integrates generalization test results and generates complete output files
"""

import os
import json
import struct
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Set
import argparse
from dataclasses import dataclass
from collections import defaultdict

@dataclass
class Instruction:
    """Represents a disassembled CHIP-8 instruction"""
    address: int
    opcode: int
    mnemonic: str
    operands: str
    description: str
    affects_registers: Set[int]
    reads_registers: Set[int]
    is_jump: bool = False
    is_call: bool = False
    is_conditional: bool = False
    target_address: Optional[int] = None

@dataclass
class SortingAnalysis:
    """Analysis results for a sorting ROM"""
    filename: str
    metadata: Dict
    instructions: List[Instruction]
    register_modifications: Dict[int, List[int]]  # register -> list of instruction addresses
    control_flow: List[Tuple[int, int]]  # (source, target) pairs
    potential_sorting_instructions: List[Instruction]
    analysis_summary: str
    authenticity: Dict = None  # Authenticity analysis results
    generalization_score: Optional[float] = None  # From generalization testing

class CHIP8Decompiler:
    """Complete CHIP-8 decompiler with sorting behavior analysis"""
    
    def __init__(self):
        self.font_area = set(range(0x50, 0x50 + 80))  # Font data area
        
    def load_generalization_results(self, search_dir: str) -> Dict[str, float]:
        """Load generalization test results to inform authenticity analysis"""
        results = {}
        search_path = Path(search_dir)
        
        # Look for generalization results files
        for results_file in search_path.rglob("generalization_analysis_*_compressed.json"):
            try:
                with open(results_file, 'r') as f:
                    data = json.load(f)
                    
                # Extract ROM scores from generalization results
                for result in data.get('results', []):
                    filename = result.get('file', '')
                    score = result.get('score', 0.0)
                    if filename:
                        results[filename] = score
                        
            except Exception as e:
                print(f"Warning: Could not load generalization results from {results_file}: {e}")
                
        return results

    def disassemble_instruction(self, address: int, opcode: int) -> Instruction:
        """
        Disassemble a single CHIP-8 instruction with EXACT behavior matching the CUDA kernel
        
        This MUST match the instruction decoding in sorting_emulator.py CUDA kernel exactly,
        or the analysis will be wrong!
        """
        
        # Extract instruction components - EXACT match to CUDA kernel
        x = (opcode & 0x0F00) >> 8
        y = (opcode & 0x00F0) >> 4
        n = opcode & 0x000F
        kk = opcode & 0x00FF
        nnn = opcode & 0x0FFF
        
        affects_registers = set()
        reads_registers = set()
        is_jump = False
        is_call = False
        is_conditional = False
        target_address = None
        
        # Decode EXACTLY as the CUDA kernel does - using switch on (opcode & 0xF000) >> 12
        opcode_high = (opcode & 0xF000) >> 12
        
        if opcode_high == 0x0:
            if opcode == 0x00E0:
                mnemonic = "CLS"
                operands = ""
                description = "Clear display"
            elif opcode == 0x00EE:
                mnemonic = "RET"
                operands = ""
                description = "Return from subroutine"
            else:
                mnemonic = "NOP"
                operands = f"${opcode:04X}"
                description = "No operation (ignored by CUDA kernel)"
                
        elif opcode_high == 0x1:
            # JP addr - Jump to address
            mnemonic = "JP"
            operands = f"${nnn:03X}"
            description = f"Jump to address ${nnn:03X}"
            is_jump = True
            target_address = nnn
            
        elif opcode_high == 0x2:
            # CALL addr - Call subroutine
            mnemonic = "CALL"
            operands = f"${nnn:03X}"
            description = f"Call subroutine at ${nnn:03X}"
            is_call = True
            target_address = nnn
            
        elif opcode_high == 0x3:
            # SE Vx, byte - Skip if Vx == byte
            mnemonic = "SE"
            operands = f"V{x:X}, #{kk:02X}"
            description = f"Skip next instruction if V{x:X} == ${kk:02X}"
            reads_registers.add(x)
            is_conditional = True
            
        elif opcode_high == 0x4:
            # SNE Vx, byte - Skip if Vx != byte
            mnemonic = "SNE"
            operands = f"V{x:X}, #{kk:02X}"
            description = f"Skip next instruction if V{x:X} != ${kk:02X}"
            reads_registers.add(x)
            is_conditional = True
            
        elif opcode_high == 0x5:
            # SE Vx, Vy - Skip if Vx == Vy (only if n == 0)
            if n == 0:
                mnemonic = "SE"
                operands = f"V{x:X}, V{y:X}"
                description = f"Skip next instruction if V{x:X} == V{y:X}"
                reads_registers.update([x, y])
                is_conditional = True
            else:
                mnemonic = "UNKNOWN"
                operands = f"${opcode:04X}"
                description = "Unknown 5xxx instruction (crashes in CUDA)"
                
        elif opcode_high == 0x6:
            # LD Vx, byte - Load byte into Vx
            mnemonic = "LD"
            operands = f"V{x:X}, #{kk:02X}"
            description = f"Load ${kk:02X} into V{x:X}"
            affects_registers.add(x)
            
        elif opcode_high == 0x7:
            # ADD Vx, byte - Add byte to Vx (with 8-bit wraparound)
            mnemonic = "ADD"
            operands = f"V{x:X}, #{kk:02X}"
            description = f"Add ${kk:02X} to V{x:X} (V{x:X} = (V{x:X} + ${kk:02X}) & 0xFF)"
            affects_registers.add(x)
            reads_registers.add(x)
            
        elif opcode_high == 0x8:
            # Register operations - EXACT match to CUDA kernel
            if n == 0x0:
                # LD Vx, Vy
                mnemonic = "LD"
                operands = f"V{x:X}, V{y:X}"
                description = f"Load V{y:X} into V{x:X}"
                affects_registers.add(x)
                reads_registers.add(y)
                
            elif n == 0x1:
                # OR Vx, Vy - CUDA kernel sets VF = 0!
                mnemonic = "OR"
                operands = f"V{x:X}, V{y:X}"
                description = f"V{x:X} = V{x:X} OR V{y:X}, VF = 0"
                affects_registers.update([x, 0xF])
                reads_registers.update([x, y])
                
            elif n == 0x2:
                # AND Vx, Vy - CUDA kernel sets VF = 0!
                mnemonic = "AND"
                operands = f"V{x:X}, V{y:X}"
                description = f"V{x:X} = V{x:X} AND V{y:X}, VF = 0"
                affects_registers.update([x, 0xF])
                reads_registers.update([x, y])
                
            elif n == 0x3:
                # XOR Vx, Vy - CUDA kernel sets VF = 0!
                mnemonic = "XOR"
                operands = f"V{x:X}, V{y:X}"
                description = f"V{x:X} = V{x:X} XOR V{y:X}, VF = 0"
                affects_registers.update([x, 0xF])
                reads_registers.update([x, y])
                
            elif n == 0x4:
                # ADD Vx, Vy - with carry flag
                mnemonic = "ADD"
                operands = f"V{x:X}, V{y:X}"
                description = f"V{x:X} = (V{x:X} + V{y:X}) & 0xFF, VF = carry"
                affects_registers.update([x, 0xF])
                reads_registers.update([x, y])
                
            elif n == 0x5:
                # SUB Vx, Vy - V{x:X} = V{x:X} - V{y:X}
                mnemonic = "SUB"
                operands = f"V{x:X}, V{y:X}"
                description = f"V{x:X} = (V{x:X} - V{y:X}) & 0xFF, VF = NOT borrow"
                affects_registers.update([x, 0xF])
                reads_registers.update([x, y])
                
            elif n == 0x6:
                # SHR Vx - Shift right
                mnemonic = "SHR"
                operands = f"V{x:X}"
                description = f"V{x:X} = V{x:X} >> 1, VF = LSB"
                affects_registers.update([x, 0xF])
                reads_registers.add(x)
                
            elif n == 0x7:
                # SUBN Vx, Vy - V{x:X} = V{y:X} - V{x:X}
                mnemonic = "SUBN"
                operands = f"V{x:X}, V{y:X}"
                description = f"V{x:X} = (V{y:X} - V{x:X}) & 0xFF, VF = NOT borrow"
                affects_registers.update([x, 0xF])
                reads_registers.update([x, y])
                
            elif n == 0xE:
                # SHL Vx - Shift left
                mnemonic = "SHL"
                operands = f"V{x:X}"
                description = f"V{x:X} = (V{x:X} << 1) & 0xFF, VF = MSB"
                affects_registers.update([x, 0xF])
                reads_registers.add(x)
                
            else:
                mnemonic = "UNKNOWN"
                operands = f"${opcode:04X}"
                description = "Unknown 8xxx instruction (crashes in CUDA)"
                
        elif opcode_high == 0x9:
            # SNE Vx, Vy - Skip if Vx != Vy (only if n == 0)
            if n == 0:
                mnemonic = "SNE"
                operands = f"V{x:X}, V{y:X}"
                description = f"Skip next instruction if V{x:X} != V{y:X}"
                reads_registers.update([x, y])
                is_conditional = True
            else:
                mnemonic = "UNKNOWN"
                operands = f"${opcode:04X}"
                description = "Unknown 9xxx instruction (crashes in CUDA)"
                
        elif opcode_high == 0xA:
            # LD I, addr - Load address into I
            mnemonic = "LD"
            operands = f"I, ${nnn:03X}"
            description = f"Load ${nnn:03X} into I register"
            
        elif opcode_high == 0xB:
            # JP V0, addr - Jump to V0 + addr
            mnemonic = "JP"
            operands = f"V0, ${nnn:03X}"
            description = f"Jump to ${nnn:03X} + V0"
            reads_registers.add(0)
            is_jump = True
            
        elif opcode_high == 0xC:
            # RND Vx, byte - Random number AND byte
            mnemonic = "RND"
            operands = f"V{x:X}, #{kk:02X}"
            description = f"V{x:X} = random() AND ${kk:02X}"
            affects_registers.add(x)
            
        elif opcode_high == 0xD:
            # DRW Vx, Vy, nibble - Draw sprite
            mnemonic = "DRW"
            operands = f"V{x:X}, V{y:X}, #{n:X}"
            description = f"Draw {n}-byte sprite at (V{x:X}, V{y:X}), VF = collision"
            reads_registers.update([x, y])
            affects_registers.add(0xF)
            
        elif opcode_high == 0xE:
            # Key operations
            if kk == 0x9E:
                mnemonic = "SKP"
                operands = f"V{x:X}"
                description = f"Skip next instruction if key V{x:X} is pressed"
                reads_registers.add(x)
                is_conditional = True
            elif kk == 0xA1:
                mnemonic = "SKNP"
                operands = f"V{x:X}"
                description = f"Skip next instruction if key V{x:X} is not pressed"
                reads_registers.add(x)
                is_conditional = True
            else:
                mnemonic = "UNKNOWN"
                operands = f"${opcode:04X}"
                description = "Unknown Exxx instruction (crashes in CUDA)"
                
        elif opcode_high == 0xF:
            # Misc operations - EXACT match to CUDA kernel
            if kk == 0x07:
                # LD Vx, DT
                mnemonic = "LD"
                operands = f"V{x:X}, DT"
                description = f"Load delay timer into V{x:X}"
                affects_registers.add(x)
                
            elif kk == 0x0A:
                # LD Vx, K - Wait for key (sets waiting_for_key flag in CUDA)
                mnemonic = "LD"
                operands = f"V{x:X}, K"
                description = f"Wait for key press, store key in V{x:X} (HALTS until key)"
                affects_registers.add(x)
                
            elif kk == 0x15:
                # LD DT, Vx
                mnemonic = "LD"
                operands = f"DT, V{x:X}"
                description = f"Load V{x:X} into delay timer"
                reads_registers.add(x)
                
            elif kk == 0x18:
                # LD ST, Vx
                mnemonic = "LD"
                operands = f"ST, V{x:X}"
                description = f"Load V{x:X} into sound timer"
                reads_registers.add(x)
                
            elif kk == 0x1E:
                # ADD I, Vx - CUDA kernel uses 16-bit wraparound
                mnemonic = "ADD"
                operands = f"I, V{x:X}"
                description = f"I = (I + V{x:X}) & 0xFFFF"
                reads_registers.add(x)
                
            elif kk == 0x29:
                # LD F, Vx - Set I to font location (CUDA: I = 0x50 + (V{x:X} & 0xF) * 5)
                mnemonic = "LD"
                operands = f"F, V{x:X}"
                description = f"I = font_address(V{x:X} & 0xF) = 0x50 + (V{x:X} & 0xF) * 5"
                reads_registers.add(x)
                
            elif kk == 0x33:
                # LD B, Vx - Store BCD
                mnemonic = "LD"
                operands = f"B, V{x:X}"
                description = f"Store BCD of V{x:X} at [I], [I+1], [I+2]"
                reads_registers.add(x)
                
            elif kk == 0x55:
                # LD [I], Vx - Store registers V0 through Vx (CUDA: I += x + 1)
                mnemonic = "LD"
                operands = f"[I], V{x:X}"
                description = f"Store V0-V{x:X} at [I], then I += {x+1}"
                reads_registers.update(range(x + 1))
                
            elif kk == 0x65:
                # LD Vx, [I] - Load registers V0 through Vx (CUDA: I += x + 1)
                mnemonic = "LD"
                operands = f"V{x:X}, [I]"
                description = f"Load V0-V{x:X} from [I], then I += {x+1}"
                affects_registers.update(range(x + 1))
                
            else:
                mnemonic = "UNKNOWN"
                operands = f"${opcode:04X}"
                description = "Unknown Fxxx instruction (crashes in CUDA)"
                
        else:
            # This should never happen since we covered 0x0-0xF
            mnemonic = "UNKNOWN"
            operands = f"${opcode:04X}"
            description = "Unknown instruction (crashes in CUDA)"
        
        return Instruction(
            address=address,
            opcode=opcode,
            mnemonic=mnemonic,
            operands=operands,
            description=description,
            affects_registers=affects_registers,
            reads_registers=reads_registers,
            is_jump=is_jump,
            is_call=is_call,
            is_conditional=is_conditional,
            target_address=target_address
        )
    
    def disassemble_rom(self, rom_data: bytes) -> List[Instruction]:
        """
        Disassemble entire ROM starting from 0x200 - EXACT match to CUDA kernel execution
        """
        instructions = []
        
        # CUDA kernel constants - MUST match sorting_emulator.py
        PROGRAM_START = 0x200
        MEMORY_SIZE = 4096
        FONT_START = 0x50
        FONT_SIZE = 80
        
        # Font area addresses (where CHIP8_FONT is loaded)
        font_area = set(range(FONT_START, FONT_START + FONT_SIZE))
        
        # Process the ROM data exactly as it would be in CUDA memory
        # ROM data gets loaded starting at PROGRAM_START (0x200)
        max_address = min(PROGRAM_START + len(rom_data), MEMORY_SIZE)
        
        # Process 2 bytes at a time (CHIP-8 instructions are 16-bit big-endian)
        for i in range(0, len(rom_data) - 1, 2):
            address = PROGRAM_START + i
            
            # Bounds check - same as CUDA kernel
            if address >= MEMORY_SIZE - 1:  # Need 2 bytes for instruction
                break
                
            # Read 16-bit big-endian instruction - EXACT match to CUDA:
            high_byte = rom_data[i]
            low_byte = rom_data[i + 1]
            opcode = (high_byte << 8) | low_byte
            
            # Skip font area addresses (font data, not program code)
            if address in font_area:
                continue
                
            instruction = self.disassemble_instruction(address, opcode)
            instructions.append(instruction)
        
        return instructions
    
    def analyze_register_flow(self, instructions: List[Instruction]) -> Dict[int, List[int]]:
        """Analyze which instructions modify each register (focus on V0-V7)"""
        register_modifications = defaultdict(list)
        
        for instr in instructions:
            for reg in instr.affects_registers:
                if 0 <= reg <= 7:  # Focus on sorting registers V0-V7
                    register_modifications[reg].append(instr.address)
        
        return dict(register_modifications)
    
    def analyze_control_flow(self, instructions: List[Instruction]) -> List[Tuple[int, int]]:
        """Analyze control flow (jumps, calls, branches)"""
        control_flow = []
        
        for instr in instructions:
            if instr.target_address is not None:
                control_flow.append((instr.address, instr.target_address))
        
        return control_flow
    
    def analyze_sorting_authenticity(self, analysis: SortingAnalysis) -> Dict:
        """
        FIXED: Enhanced authenticity analysis that properly considers generalization results
        """
        metadata = analysis.metadata
        sort_info = metadata.get('partial_sorting', {})
        initial_state = metadata.get('registers', {}).get('initial', [])
        final_state = metadata.get('registers', {}).get('final', [])
        sequence = sort_info.get('sequence', [])
        generalization_score = analysis.generalization_score
        
        authenticity = {
            'is_genuine_sorting': False,
            'confidence': 0.0,
            'evidence': [],
            'red_flags': [],
            'classification': 'UNKNOWN'
        }
        
        # CRITICAL: Generalization score is the PRIMARY evidence
        if generalization_score is not None:
            if generalization_score >= 0.70:  # 70%+ success rate
                authenticity['evidence'].append(f"EXCELLENT generalization: {generalization_score:.1%} success across test patterns")
                authenticity['confidence'] += 0.8  # Very strong evidence
            elif generalization_score >= 0.50:  # 50-70% success rate
                authenticity['evidence'].append(f"GOOD generalization: {generalization_score:.1%} success across test patterns")
                authenticity['confidence'] += 0.6  # Strong evidence
            elif generalization_score >= 0.30:  # 30-50% success rate
                authenticity['evidence'].append(f"MODERATE generalization: {generalization_score:.1%} success across test patterns")
                authenticity['confidence'] += 0.4  # Moderate evidence
            elif generalization_score >= 0.15:  # 15-30% success rate
                authenticity['evidence'].append(f"WEAK generalization: {generalization_score:.1%} success across test patterns")
                authenticity['confidence'] += 0.2  # Weak evidence
            else:
                authenticity['red_flags'].append(f"POOR generalization: only {generalization_score:.1%} success across test patterns")
        
        # LESS CRITICAL: Static pattern analysis (now secondary)
        expected_initial = [8, 3, 6, 1, 7, 2, 5, 4]
        if initial_state != expected_initial:
            authenticity['red_flags'].append(f"Wrong initial state: {initial_state} (expected {expected_initial})")
        
        # Only flag consecutive sequences as suspicious if generalization is poor
        if sequence and generalization_score is not None and generalization_score < 0.30:
            if len(sequence) > 2:
                is_consecutive = all(sequence[i] + 1 == sequence[i + 1] for i in range(len(sequence) - 1))
                if is_consecutive and sequence[0] not in expected_initial:
                    authenticity['red_flags'].append(f"Poor generalization + perfect consecutive sequence suggests coincidence")
        
        # Check for excessive random number generation (still relevant)
        rnd_instructions = [instr for instr in analysis.instructions if instr.mnemonic == 'RND']
        if len(rnd_instructions) > 50:  # Raised threshold
            authenticity['red_flags'].append(f"Dominated by random generation: {len(rnd_instructions)} RND instructions")
        
        # POSITIVE EVIDENCE: Look for genuine sorting patterns
        sorted_regs = set(range(sort_info.get('start_position', 0), 
                               sort_info.get('start_position', 0) + sort_info.get('length', 0)))

        register_comparisons = 0
        register_transfers = 0
        
        for instr in analysis.instructions:
            if instr.mnemonic in ['SE', 'SNE'] and len(instr.reads_registers) == 2:
                # Comparing two registers
                if instr.reads_registers.issubset(sorted_regs | {15}):  # Include VF
                    register_comparisons += 1
            
            if instr.mnemonic == 'LD' and len(instr.reads_registers) == 1 and len(instr.affects_registers) == 1:
                # Register to register copy
                if (instr.reads_registers | instr.affects_registers).issubset(sorted_regs):
                    register_transfers += 1
        
        if register_comparisons >= 3:
            authenticity['evidence'].append(f"Found {register_comparisons} register comparisons in sorted range")
        
        if register_transfers >= 3:  # Lowered threshold
            authenticity['evidence'].append(f"Found {register_transfers} register-to-register transfers")
        
        # Relationship between initial and final values
        if initial_state and final_state and sequence:
            initial_set = set(initial_state)
            final_set = set(final_state)
            common_values = initial_set & final_set
            
            if len(common_values) >= 3:
                authenticity['evidence'].append(f"Found {len(common_values)} values preserved from initial state")
        
        # FIXED: Final classification heavily weights generalization performance
        red_flag_penalty = len(authenticity['red_flags']) * 0.1  # Reduced penalty
        evidence_bonus = len(authenticity['evidence']) * 0.2    # Reduced bonus
        
        authenticity['confidence'] = max(0.0, min(1.0, authenticity['confidence'] + evidence_bonus - red_flag_penalty))
        
        # Classification based primarily on generalization score
        if generalization_score is not None:
            if generalization_score >= 0.70:
                authenticity['classification'] = 'GENUINE'
                authenticity['is_genuine_sorting'] = True
            elif generalization_score >= 0.50:
                authenticity['classification'] = 'LIKELY_GENUINE'
                authenticity['is_genuine_sorting'] = True
            elif generalization_score >= 0.30:
                authenticity['classification'] = 'PARTIAL_GENUINE'
                authenticity['is_genuine_sorting'] = False
            elif generalization_score >= 0.15:
                authenticity['classification'] = 'WEAK_GENERALIZATION'
                authenticity['is_genuine_sorting'] = False
            else:
                authenticity['classification'] = 'COINCIDENTAL'
                authenticity['is_genuine_sorting'] = False
        else:
            # Fallback to old logic if no generalization data
            if len(authenticity['red_flags']) >= 3:
                authenticity['classification'] = 'COINCIDENTAL'
                authenticity['is_genuine_sorting'] = False
            elif authenticity['confidence'] > 0.6:
                authenticity['classification'] = 'LIKELY_GENUINE'
                authenticity['is_genuine_sorting'] = True
            else:
                authenticity['classification'] = 'UNCERTAIN'
                authenticity['is_genuine_sorting'] = False
        
        return authenticity
    
    def identify_sorting_patterns(self, instructions: List[Instruction], metadata: Dict) -> List[Instruction]:
        """Identify instructions that could be related to sorting behavior"""
        potential_sorting = []
        
        # Get the sorted register range from metadata
        sort_info = metadata.get('partial_sorting', {})
        start_pos = sort_info.get('start_position', 0)
        length = sort_info.get('length', 0)
        sort_cycle = metadata.get('discovery_info', {}).get('sort_cycle', 0)
        sorted_registers = set(range(start_pos, start_pos + length))
        
        print(f"    Analyzing sorting in registers V{start_pos}-V{start_pos + length - 1}")
        print(f"    Sorting achieved at cycle {sort_cycle}")
        
        for instr in instructions:
            is_sorting_related = False
            reason = ""
            
            # PRIORITY 1: Instructions that directly modify the sorted registers
            if instr.affects_registers.intersection(sorted_registers):
                is_sorting_related = True
                modified_regs = sorted(instr.affects_registers.intersection(sorted_registers))
                reason = f"Modifies sorted registers V{modified_regs}"
                
            # PRIORITY 2: Comparison operations (could be part of sorting logic)
            elif instr.mnemonic in ['SE', 'SNE'] and instr.reads_registers.intersection(sorted_registers):
                is_sorting_related = True
                read_regs = sorted(instr.reads_registers.intersection(sorted_registers))
                reason = f"Compares sorted registers V{read_regs}"
                
            # PRIORITY 3: Arithmetic operations that read sorted registers
            elif instr.mnemonic in ['ADD', 'SUB', 'SUBN'] and instr.reads_registers.intersection(sorted_registers):
                is_sorting_related = True
                read_regs = sorted(instr.reads_registers.intersection(sorted_registers))
                reason = f"Arithmetic on sorted registers V{read_regs}"
                
            # PRIORITY 4: Register-to-register copies involving sorted registers
            elif (instr.mnemonic == 'LD' and 
                  any(reg in sorted_registers for reg in instr.affects_registers | instr.reads_registers)):
                is_sorting_related = True
                all_regs = sorted((instr.affects_registers | instr.reads_registers).intersection(sorted_registers))
                reason = f"Register transfer involving V{all_regs}"
                
            # PRIORITY 5: Logical operations (OR/AND/XOR set VF=0 in CUDA!)
            elif (instr.mnemonic in ['OR', 'AND', 'XOR'] and 
                  instr.reads_registers.intersection(sorted_registers)):
                is_sorting_related = True
                read_regs = sorted(instr.reads_registers.intersection(sorted_registers))
                reason = f"Logical operation on V{read_regs} (sets VF=0 in CUDA)"
                
            # PRIORITY 6: Shifts on sorted registers
            elif (instr.mnemonic in ['SHR', 'SHL'] and 
                  instr.reads_registers.intersection(sorted_registers)):
                is_sorting_related = True
                read_regs = sorted(instr.reads_registers.intersection(sorted_registers))
                reason = f"Bit shift on V{read_regs}"
                
            # PRIORITY 7: Bulk register operations (F55/F65) that include sorted registers
            elif (instr.mnemonic == 'LD' and '[I]' in instr.operands and
                  any(reg in sorted_registers for reg in instr.affects_registers | instr.reads_registers)):
                is_sorting_related = True
                reason = "Bulk register load/store affecting sorted registers"
                
            # PRIORITY 8: Random number generation affecting sorted registers
            elif instr.mnemonic == 'RND' and instr.affects_registers.intersection(sorted_registers):
                is_sorting_related = True
                modified_regs = sorted(instr.affects_registers.intersection(sorted_registers))
                reason = f"Random number into sorted register V{modified_regs}"
            
            if is_sorting_related:
                # Add metadata about why this instruction is interesting
                instr.sorting_reason = reason
                potential_sorting.append(instr)
        
        print(f"    Found {len(potential_sorting)} potentially sorting-related instructions")
        return potential_sorting
    
    def generate_analysis_summary(self, analysis: SortingAnalysis) -> str:
        """Generate a comprehensive analysis summary with enhanced generalization focus"""
        metadata = analysis.metadata
        sort_info = metadata.get('partial_sorting', {})
        authenticity = getattr(analysis, 'authenticity', {})
        
        summary = []
        summary.append(f"=== SORTING ROM ANALYSIS: {analysis.filename} ===")
        summary.append("")
        
        # AUTHENTICITY ASSESSMENT (CRITICAL)
        if authenticity:
            confidence = authenticity.get('confidence', 0.0)
            classification = authenticity.get('classification', 'UNKNOWN')
            is_genuine = authenticity.get('is_genuine_sorting', False)
            
            summary.append("🔍 SORTING AUTHENTICITY ASSESSMENT:")
            if is_genuine:
                summary.append(f"  ✅ CLASSIFICATION: {classification} (Confidence: {confidence:.1%})")
            else:
                summary.append(f"  ❌ CLASSIFICATION: {classification} (Confidence: {confidence:.1%})")
            
            # Show generalization score prominently
            if analysis.generalization_score is not None:
                summary.append(f"  📊 GENERALIZATION SCORE: {analysis.generalization_score:.1%} success across test patterns")
            
            # Show evidence
            evidence = authenticity.get('evidence', [])
            red_flags = authenticity.get('red_flags', [])
            
            if evidence:
                summary.append("  Evidence for genuine sorting:")
                for ev in evidence:
                    summary.append(f"    ✓ {ev}")
            
            if red_flags:
                summary.append("  Red flags against genuine sorting:")
                for flag in red_flags:
                    summary.append(f"    ⚠️  {flag}")
            
            summary.append("")
            
            # Early warning if this is clearly fake
            if not is_genuine and analysis.generalization_score is not None and analysis.generalization_score < 0.30:
                summary.append("⚠️  WARNING: Poor generalization suggests COINCIDENTAL consecutive values, NOT genuine sorting!")
                summary.append("   The sorted output may be from random code rather than algorithmic transformation.")
                summary.append("")
        
        # Rest of the analysis remains the same...
        summary.append("SORTING ACHIEVEMENT:")
        initial_pattern = metadata.get('registers', {}).get('initial', [])
        final_pattern = metadata.get('registers', {}).get('final', [])
        sequence = sort_info.get('sequence', [])
        
        summary.append(f"  Sorted sequence: {sequence}")
        summary.append(f"  Length: {sort_info.get('length', 0)} consecutive elements")
        summary.append(f"  Direction: {sort_info.get('direction', 'Unknown')}")
        summary.append(f"  Register range: {sort_info.get('sequence_range', 'Unknown')}")
        summary.append(f"  Start position: V{sort_info.get('start_position', 0)}")
        summary.append(f"  End position: V{sort_info.get('end_position', 0)}")
        
        # Show the transformation
        if initial_pattern and final_pattern:
            summary.append(f"  Initial state: {initial_pattern}")
            summary.append(f"  Final state:   {final_pattern}")
            
            # Highlight which positions changed
            changes = []
            for i, (init, final) in enumerate(zip(initial_pattern, final_pattern)):
                if init != final:
                    changes.append(f"V{i}: {init}→{final}")
            if changes:
                summary.append(f"  Changes: {', '.join(changes)}")
            
            # Check if this looks like genuine sorting vs random replacement
            expected_initial = [8, 3, 6, 1, 7, 2, 5, 4]
            if initial_pattern == expected_initial:
                summary.append("  ✓ Correct initial test pattern detected")
            else:
                summary.append(f"  ❌ Wrong initial pattern! Expected {expected_initial}")
        
        summary.append(f"  Achievement cycle: {metadata.get('discovery_info', {}).get('sort_cycle', 'Unknown')}")
        summary.append("")
        
        return "\n".join(summary)
    
    def generate_complete_disassembly(self, analysis: SortingAnalysis) -> str:
        """Generate complete disassembly with full details for file output"""
        lines = []
        
        # Header with ROM information
        lines.append("=" * 80)
        lines.append(f"COMPLETE CHIP-8 SORTING ROM ANALYSIS")
        lines.append(f"ROM: {analysis.filename}")
        lines.append(f"Analysis Date: {__import__('datetime').datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append("=" * 80)
        lines.append("")
        
        # Authenticity summary
        lines.append(analysis.analysis_summary)
        lines.append("")
        lines.append("=" * 80)
        lines.append("COMPLETE INSTRUCTION DISASSEMBLY")
        lines.append("=" * 80)
        lines.append("")
        
        # Show metadata
        sort_info = analysis.metadata.get('partial_sorting', {})
        lines.append(f"Sorting Achievement: {sort_info.get('sequence', [])} ({sort_info.get('direction', 'unknown')})")
        lines.append(f"Registers {sort_info.get('sequence_range', 'unknown')} sorted at cycle {analysis.metadata.get('discovery_info', {}).get('sort_cycle', 'unknown')}")
        
        if analysis.generalization_score is not None:
            lines.append(f"Generalization Score: {analysis.generalization_score:.1%} success across test patterns")
        
        lines.append("")
        lines.append("INSTRUCTION LISTING:")
        lines.append("ADDRESS  OPCODE  MNEMONIC OPERANDS         DESCRIPTION                                      REGISTERS")
        lines.append("-" * 120)
        
        # Complete disassembly with enhanced information
        for instr in analysis.instructions:
            # Mark potential sorting instructions
            marker = ">>> " if instr in analysis.potential_sorting_instructions else "    "
            
            # Register information
            reg_info = ""
            if instr.affects_registers or instr.reads_registers:
                affects = f"W:{sorted(instr.affects_registers)}" if instr.affects_registers else ""
                reads = f"R:{sorted(instr.reads_registers)}" if instr.reads_registers else ""
                reg_info = f"{affects} {reads}".strip()
            
            # Format instruction line
            line = f"{marker}${instr.address:03X}    ${instr.opcode:04X}   {instr.mnemonic:8} {instr.operands:15} {instr.description:45} {reg_info}"
            lines.append(line)
            
            # Show sorting relevance for important instructions
            if instr in analysis.potential_sorting_instructions:
                reason = getattr(instr, 'sorting_reason', 'Affects sorted registers')
                lines.append(f"          ^-- SORTING RELATED: {reason}")
        
        lines.append("")
        lines.append("=" * 80)
        lines.append("REGISTER FLOW ANALYSIS")
        lines.append("=" * 80)
        lines.append("")
        
        # Register modification analysis
        if analysis.register_modifications:
            lines.append("Register modifications (V0-V7 focus):")
            for reg in range(8):
                if reg in analysis.register_modifications:
                    addresses = analysis.register_modifications[reg]
                    lines.append(f"  V{reg}: Modified at {len(addresses)} locations")
                    addr_list = [f"${addr:03X}" for addr in addresses]
                    lines.append(f"       Addresses: {', '.join(addr_list)}")
                else:
                    lines.append(f"  V{reg}: Not modified")
        else:
            lines.append("No register modifications detected in V0-V7")
        
        lines.append("")
        lines.append("=" * 80)
        lines.append("CONTROL FLOW ANALYSIS")
        lines.append("=" * 80)
        lines.append("")
        
        # Control flow analysis
        if analysis.control_flow:
            lines.append(f"{len(analysis.control_flow)} control transfers detected:")
            for source, target in analysis.control_flow:
                is_loop = target <= source
                loop_indicator = " (LOOP)" if is_loop else " (FORWARD)"
                lines.append(f"  ${source:03X} → ${target:03X}{loop_indicator}")
        else:
            lines.append("No jumps or calls detected (linear execution)")
        
        lines.append("")
        lines.append("=" * 80)
        lines.append("SORTING INSTRUCTION ANALYSIS")
        lines.append("=" * 80)
        lines.append("")
        
        # Sorting instruction breakdown
        if analysis.potential_sorting_instructions:
            lines.append(f"{len(analysis.potential_sorting_instructions)} instructions identified as sorting-related:")
            lines.append("")
            
            # Group by type
            by_type = {}
            for instr in analysis.potential_sorting_instructions:
                reason = getattr(instr, 'sorting_reason', 'Unknown reason')
                if reason not in by_type:
                    by_type[reason] = []
                by_type[reason].append(instr)
            
            # Show each category in detail
            for reason, instrs in by_type.items():
                lines.append(f"{reason}: {len(instrs)} instructions")
                for instr in instrs:
                    reg_info = ""
                    if instr.affects_registers or instr.reads_registers:
                        affects = f"Writes:{sorted(instr.affects_registers)}" if instr.affects_registers else ""
                        reads = f"Reads:{sorted(instr.reads_registers)}" if instr.reads_registers else ""
                        reg_info = f" [{affects} {reads}]".strip()
                    
                    lines.append(f"  ${instr.address:03X}: {instr.mnemonic:4} {instr.operands:12} ; {instr.description}{reg_info}")
                lines.append("")
        else:
            lines.append("No sorting-related instructions identified")
        
        lines.append("=" * 80)
        lines.append("END OF ANALYSIS")
        lines.append("=" * 80)
        
        return "\n".join(lines)
    
    def analyze_sorting_rom(self, rom_path: Path, metadata_path: Path, generalization_results: Dict[str, float] = None) -> SortingAnalysis:
        """Complete analysis of a sorting ROM with generalization integration"""
        
        # Load ROM binary
        with open(rom_path, 'rb') as f:
            rom_data = f.read()
        
        # Load metadata
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        # Disassemble
        instructions = self.disassemble_rom(rom_data)
        
        # Analyze
        register_modifications = self.analyze_register_flow(instructions)
        control_flow = self.analyze_control_flow(instructions)
        potential_sorting = self.identify_sorting_patterns(instructions, metadata)
        
        # Create analysis object
        analysis = SortingAnalysis(
            filename=rom_path.name,
            metadata=metadata,
            instructions=instructions,
            register_modifications=register_modifications,
            control_flow=control_flow,
            potential_sorting_instructions=potential_sorting,
            analysis_summary=""
        )
        
        # Get generalization score if available
        if generalization_results and rom_path.name in generalization_results:
            analysis.generalization_score = generalization_results[rom_path.name]
        
        # CRITICAL: Analyze authenticity with generalization data
        authenticity = self.analyze_sorting_authenticity(analysis)
        analysis.authenticity = authenticity
        
        # Generate summary
        analysis.analysis_summary = self.generate_analysis_summary(analysis)
        
        return analysis


def find_sorting_roms(search_dir: str) -> List[Tuple[Path, Path]]:
    """Find all sorting ROM files and their metadata"""
    search_path = Path(search_dir)
    rom_pairs = []
    
    # Find all .ch8 files that start with LONGPARTIAL
    for rom_file in search_path.rglob("LONGPARTIAL_*.ch8"):
        # Look for corresponding .json file
        json_file = rom_file.with_suffix('.json')
        
        if json_file.exists():
            rom_pairs.append((rom_file, json_file))
        else:
            print(f"Warning: Found ROM {rom_file} but no metadata file {json_file}")
    
    return sorted(rom_pairs)

def main():
    parser = argparse.ArgumentParser(
        description="CHIP-8 Sorting ROM Decompiler - Analyze discovered sorting behavior",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python sorting_decompiler.py output/async_partial_sorting
  python sorting_decompiler.py output/async_partial_sorting --length 8
  python sorting_decompiler.py output/async_partial_sorting --rom LONGPARTIAL_B28055D01_V0-V7_L8_ASC_C282_f341c602.ch8
  python sorting_decompiler.py output/async_partial_sorting --summary-only
  python sorting_decompiler.py output/async_partial_sorting --full-disassembly --output-file complete_analysis.txt
        """
    )
    
    parser.add_argument('search_dir', 
                       help='Directory to search for sorting ROMs (e.g., output/async_partial_sorting)')
    parser.add_argument('--length', type=int, choices=[6, 7, 8],
                       help='Only analyze ROMs with specific sorting length')
    parser.add_argument('--direction', choices=['ascending', 'descending'],
                       help='Only analyze ROMs with specific sorting direction')
    parser.add_argument('--rom', type=str,
                       help='Analyze specific ROM file by name')
    parser.add_argument('--summary-only', action='store_true',
                       help='Show only analysis summaries, not full disassembly')
    parser.add_argument('--full-disassembly', action='store_true',
                       help='Show complete disassembly for each ROM')
    parser.add_argument('--output-file', type=str,
                       help='Save analysis to file instead of printing')
    
    args = parser.parse_args()
    
    print("CHIP-8 SORTING ROM DECOMPILER")
    print("=" * 50)
    print("Analyzing discovered sorting behavior in CHIP-8 ROMs")
    print()
    
    # Find all sorting ROMs
    print(f"Searching for sorting ROMs in: {args.search_dir}")
    rom_pairs = find_sorting_roms(args.search_dir)
    
    if not rom_pairs:
        print("No sorting ROMs found!")
        print("Make sure you're pointing to the correct output directory.")
        return 1
    
    print(f"Found {len(rom_pairs)} sorting ROMs")
    
    # Initialize decompiler and load generalization results
    decompiler = CHIP8Decompiler()
    print("Loading generalization test results...")
    generalization_results = decompiler.load_generalization_results(args.search_dir)
    
    if generalization_results:
        print(f"Loaded generalization scores for {len(generalization_results)} ROMs")
    else:
        print("No generalization results found - authenticity analysis will be limited")
    
    # Filter ROMs based on criteria
    filtered_pairs = []
    for rom_file, json_file in rom_pairs:
        # Load metadata for filtering
        with open(json_file, 'r') as f:
            metadata = json.load(f)
        
        # Apply filters
        if args.length:
            if metadata.get('partial_sorting', {}).get('length') != args.length:
                continue
        
        if args.direction:
            if metadata.get('partial_sorting', {}).get('direction') != args.direction:
                continue
                
        if args.rom:
            if args.rom not in rom_file.name:
                continue
        
        filtered_pairs.append((rom_file, json_file))
    
    if not filtered_pairs:
        print("No ROMs match the specified criteria!")
        return 1
    
    print(f"Analyzing {len(filtered_pairs)} ROMs matching criteria...")
    print()
    
    # Analyze each ROM
    analyses = []
    output_lines = []
    
    for i, (rom_file, json_file) in enumerate(filtered_pairs, 1):
        print(f"Analyzing ROM {i}/{len(filtered_pairs)}: {rom_file.name}")
        
        try:
            analysis = decompiler.analyze_sorting_rom(rom_file, json_file, generalization_results)
            analyses.append(analysis)
            
            # Add to output
            if args.full_disassembly:
                output_lines.append(decompiler.generate_complete_disassembly(analysis))
            else:
                output_lines.append(analysis.analysis_summary)
            
            output_lines.append("")
            output_lines.append("=" * 100)
            output_lines.append("")
        
        except Exception as e:
            print(f"Error analyzing {rom_file.name}: {e}")
            continue
    
    # Generate summary statistics
    output_lines.append("=== OVERALL ANALYSIS SUMMARY ===")
    output_lines.append("")
    output_lines.append(f"Total ROMs analyzed: {len(analyses)}")
    
    if analyses:
        # Authenticity analysis summary
        output_lines.append("")
        output_lines.append("AUTHENTICITY ANALYSIS:")
        genuine_count = sum(1 for a in analyses if getattr(a, 'authenticity', {}).get('is_genuine_sorting', False))
        coincidental_count = len(analyses) - genuine_count
        
        output_lines.append(f"  Genuine sorting algorithms: {genuine_count}")
        output_lines.append(f"  Coincidental consecutive values: {coincidental_count}")
        
        if genuine_count > 0:
            output_lines.append(f"  ✅ GENUINE ALGORITHMS FOUND!")
            output_lines.append(f"  Genuine sorting rate: {genuine_count/len(analyses):.1%}")
            
            # List the genuine ones
            genuine_roms = [a for a in analyses if getattr(a, 'authenticity', {}).get('is_genuine_sorting', False)]
            output_lines.append("")
            output_lines.append("🎉 GENUINE SORTING ALGORITHMS:")
            for analysis in genuine_roms:
                score = analysis.generalization_score
                score_text = f"{score:.1%} success rate" if score else "No generalization data"
                output_lines.append(f"   {analysis.filename}: {score_text}")
        else:
            output_lines.append("  ⚠️  NO GENUINE SORTING FOUND - All appear to be coincidental!")
    
    # Output results
    output_text = "\n".join(output_lines)
    
    if args.output_file:
        with open(args.output_file, 'w', encoding='utf-8') as f:
            f.write(output_text)
        print(f"Complete analysis saved to: {args.output_file}")
    else:
        print(output_text)
    
    print(f"\nAnalysis complete! Processed {len(analyses)} sorting ROMs.")
    
    return 0

if __name__ == "__main__":
    try:
        exit(main())
    except KeyboardInterrupt:
        print("\nAnalysis interrupted by user")
        exit(1)
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        exit(1)