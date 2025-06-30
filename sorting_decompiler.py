#!/usr/bin/env python3
"""
CHIP-8 Sorting ROM Decompiler with Authenticity Detection
Analyzes discovered sorting ROMs to determine genuine sorting vs. coincidental consecutive values

Features:
- Complete CHIP-8 disassembly with exact CUDA kernel behavioral matching
- Authenticity analysis to detect genuine sorting vs. random consecutive placement
- Register flow analysis focused on sorting registers (V0-V7)
- Control flow analysis and sorting pattern detection
- Detailed reporting on sorting mechanisms and red flags
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

class CHIP8Decompiler:
    """Complete CHIP-8 decompiler with sorting behavior analysis"""
    
    def __init__(self):
        self.font_area = set(range(0x50, 0x50 + 80))  # Font data area
        
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
        
        CRITICAL: This must process memory exactly as the CUDA kernel does:
        - Programs loaded at 0x200 (PROGRAM_START)
        - Font data at 0x50-0x9F (80 bytes)
        - Memory size is 4096 bytes total
        - Instructions are fetched as big-endian 16-bit values
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
            # const unsigned char high_byte = memory[mem_base + pc];
            # const unsigned char low_byte = memory[mem_base + pc + 1]; 
            # const unsigned short instruction = (high_byte << 8) | low_byte;
            high_byte = rom_data[i]
            low_byte = rom_data[i + 1]
            opcode = (high_byte << 8) | low_byte
            
            # Skip font area addresses (font data, not program code)
            if address in font_area:
                continue
                
            # Skip all-zero instructions (but still show them for completeness)
            # The CUDA kernel will execute these as NOP effectively
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
        Determine if this is genuine sorting vs. coincidental consecutive values
        
        CRITICAL: Many "discoveries" are just random consecutive values, not actual sorting!
        """
        metadata = analysis.metadata
        sort_info = metadata.get('partial_sorting', {})
        initial_state = metadata.get('registers', {}).get('initial', [])
        final_state = metadata.get('registers', {}).get('final', [])
        sequence = sort_info.get('sequence', [])
        
        authenticity = {
            'is_genuine_sorting': False,
            'confidence': 0.0,
            'evidence': [],
            'red_flags': [],
            'classification': 'UNKNOWN'
        }
        
        # RED FLAG 1: Initial pattern completely ignored
        expected_initial = [8, 3, 6, 1, 7, 2, 5, 4]
        if initial_state != expected_initial:
            authenticity['red_flags'].append(f"Wrong initial state: {initial_state} (expected {expected_initial})")
        
        # RED FLAG 2: Sorted values have no relationship to initial values
        if sequence:
            # Check if sorted values are transformations of initial values
            initial_in_range = any(val in range(min(sequence), max(sequence) + 1) for val in expected_initial)
            if not initial_in_range:
                authenticity['red_flags'].append(f"Sorted values {sequence} completely unrelated to initial pattern {expected_initial}")
            
            # Check if it's just consecutive integers (very suspicious)
            if len(sequence) > 2:
                is_consecutive = all(sequence[i] + 1 == sequence[i + 1] for i in range(len(sequence) - 1))
                if is_consecutive and sequence[0] not in expected_initial:
                    authenticity['red_flags'].append(f"Perfect consecutive sequence {sequence} with no initial pattern involvement")
        
        # RED FLAG 3: Too many "sorting" instructions (probably just random code)
        sorting_instruction_ratio = len(analysis.potential_sorting_instructions) / len(analysis.instructions)
        if sorting_instruction_ratio > 0.15:  # More than 15% is suspicious
            authenticity['red_flags'].append(f"Suspicious: {sorting_instruction_ratio:.1%} of instructions marked as sorting-related")
        
        # RED FLAG 4: Dominated by random number generation
        rnd_instructions = [instr for instr in analysis.instructions if instr.mnemonic == 'RND']
        if len(rnd_instructions) > 20:
            authenticity['red_flags'].append(f"Dominated by random generation: {len(rnd_instructions)} RND instructions")
        
        # RED FLAG 5: No comparison/conditional logic
        comparison_instructions = [instr for instr in analysis.instructions if instr.mnemonic in ['SE', 'SNE']]
        if len(comparison_instructions) < 5:
            authenticity['red_flags'].append("Very few comparison instructions - unlikely to be sorting logic")
        
        # POSITIVE EVIDENCE: Look for genuine sorting patterns
        
        # EVIDENCE 1: Register-to-register comparisons and swaps
        register_comparisons = 0
        register_transfers = 0
        sorted_regs = set(range(sort_info.get('start_position', 0), 
                            sort_info.get('start_position', 0) + sort_info.get('length', 0)))

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
        
        if register_transfers >= 5:
            authenticity['evidence'].append(f"Found {register_transfers} register-to-register transfers")
        
        # EVIDENCE 2: Iterative improvement (values getting closer to sorted)
        # This would require execution tracing, which we don't have
        
        # EVIDENCE 3: Relationship between initial and final values
        if initial_state and final_state and sequence:
            # Check if some initial values appear in final positions
            initial_set = set(initial_state)
            final_set = set(final_state)
            common_values = initial_set & final_set
            
            if len(common_values) >= 3:
                authenticity['evidence'].append(f"Found {len(common_values)} values preserved from initial state")
        
        # Calculate confidence and classification
        red_flag_penalty = len(authenticity['red_flags']) * 0.2
        evidence_bonus = len(authenticity['evidence']) * 0.3
        
        authenticity['confidence'] = max(0.0, min(1.0, evidence_bonus - red_flag_penalty))
        
        if len(authenticity['red_flags']) >= 3:
            authenticity['classification'] = 'COINCIDENTAL'
            authenticity['is_genuine_sorting'] = False
        elif len(authenticity['red_flags']) <= 1 and len(authenticity['evidence']) >= 2:
            authenticity['classification'] = 'GENUINE'
            authenticity['is_genuine_sorting'] = True
        elif authenticity['confidence'] > 0.6:
            authenticity['classification'] = 'LIKELY_GENUINE'
            authenticity['is_genuine_sorting'] = True
        elif authenticity['confidence'] < 0.3:
            authenticity['classification'] = 'LIKELY_COINCIDENTAL'
            authenticity['is_genuine_sorting'] = False
        else:
            authenticity['classification'] = 'UNCERTAIN'
            authenticity['is_genuine_sorting'] = False
        
        return authenticity
    
    def identify_sorting_patterns(self, instructions: List[Instruction], metadata: Dict) -> List[Instruction]:
        """
        Identify instructions that could be related to sorting behavior
        
        ENHANCED: Now considers the exact CUDA kernel behavior and timing
        """
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
        """Generate a comprehensive analysis summary with CUDA-specific insights"""
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
            if not is_genuine:
                summary.append("⚠️  WARNING: This appears to be COINCIDENTAL consecutive values, NOT genuine sorting!")
                summary.append("   The initial test pattern [8,3,6,1,7,2,5,4] was likely overwritten with random consecutive numbers.")
                summary.append("")
        
        # Sorting achievement with enhanced details
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
        
        # CUDA-specific execution details
        register_activity = metadata.get('register_activity', {})
        summary.append("CUDA EXECUTION STATISTICS:")
        summary.append(f"  Total register operations: {register_activity.get('total_register_ops', 'Unknown')}")
        summary.append(f"  Register reads: {register_activity.get('register_reads', 'Unknown')}")
        summary.append(f"  Register writes: {register_activity.get('register_writes', 'Unknown')}")
        
        # Calculate efficiency metrics
        sort_cycle = metadata.get('discovery_info', {}).get('sort_cycle', 0)
        if sort_cycle > 0 and register_activity.get('register_writes', 0) > 0:
            writes_per_cycle = register_activity.get('register_writes', 0) / sort_cycle
            summary.append(f"  Register writes per cycle: {writes_per_cycle:.3f}")
        summary.append("")
        
        # Register modification analysis
        summary.append("REGISTER MODIFICATIONS (V0-V7 focus):")
        if analysis.register_modifications:
            for reg in range(8):  # Focus on V0-V7
                if reg in analysis.register_modifications:
                    addresses = analysis.register_modifications[reg]
                    summary.append(f"  V{reg}: Modified at {len(addresses)} locations")
                    summary.append(f"       Addresses: {[f'${addr:03X}' for addr in addresses[:8]]}")
                    if len(addresses) > 8:
                        summary.append(f"       ... and {len(addresses) - 8} more")
                else:
                    summary.append(f"  V{reg}: Not modified")
        else:
            summary.append("  No register modifications detected in V0-V7")
        summary.append("")
        
        # Control flow analysis
        summary.append("CONTROL FLOW ANALYSIS:")
        if analysis.control_flow:
            summary.append(f"  {len(analysis.control_flow)} control transfers detected:")
            for i, (source, target) in enumerate(analysis.control_flow[:5]):
                # Determine if this creates a loop
                is_loop = target <= source
                loop_indicator = " (LOOP)" if is_loop else ""
                summary.append(f"    ${source:03X} → ${target:03X}{loop_indicator}")
            if len(analysis.control_flow) > 5:
                summary.append(f"    ... and {len(analysis.control_flow) - 5} more transfers")
            
            # Look for potential sorting loops
            sorting_loops = 0
            for source, target in analysis.control_flow:
                if target <= source:  # Backward jump = potential loop
                    sorting_loops += 1
            if sorting_loops > 0:
                summary.append(f"  Potential sorting loops: {sorting_loops} backward jumps detected")
        else:
            summary.append("  No jumps or calls detected (linear execution)")
        summary.append("")
        
        # Enhanced sorting instruction analysis
        summary.append("SORTING-RELATED INSTRUCTIONS:")
        if analysis.potential_sorting_instructions:
            summary.append(f"  {len(analysis.potential_sorting_instructions)} instructions identified as sorting-related:")
            
            # Group by type for better analysis
            by_type = {}
            for instr in analysis.potential_sorting_instructions:
                reason = getattr(instr, 'sorting_reason', 'Unknown reason')
                if reason not in by_type:
                    by_type[reason] = []
                by_type[reason].append(instr)
            
            # Show each category
            for reason, instrs in by_type.items():
                summary.append(f"    {reason}: {len(instrs)} instructions")
                for instr in instrs[:3]:  # Show first 3 of each type
                    summary.append(f"      ${instr.address:03X}: {instr.mnemonic:4} {instr.operands:12} ; {instr.description}")
                if len(instrs) > 3:
                    summary.append(f"      ... and {len(instrs) - 3} more")
        else:
            summary.append("  No sorting-related instructions identified")
            summary.append("  This suggests the sorting may be emergent from complex interactions")
        summary.append("")
        
        # Code complexity and structure
        summary.append("CODE STRUCTURE ANALYSIS:")
        summary.append(f"  Total instructions: {len(analysis.instructions)}")
        summary.append(f"  ROM size: {metadata.get('rom_info', {}).get('size_bytes', 'Unknown')} bytes")
        summary.append(f"  Code density: {len(analysis.instructions) / (metadata.get('rom_info', {}).get('size_bytes', 1) / 2):.1%} of possible instruction slots")
        
        # Instruction type distribution
        instruction_types = {}
        for instr in analysis.instructions:
            if instr.mnemonic not in instruction_types:
                instruction_types[instr.mnemonic] = 0
            instruction_types[instr.mnemonic] += 1
        
        summary.append("  Instruction distribution:")
        for mnemonic, count in sorted(instruction_types.items(), key=lambda x: x[1], reverse=True)[:10]:
            percentage = (count / len(analysis.instructions)) * 100
            summary.append(f"    {mnemonic}: {count} ({percentage:.1f}%)")
        
        # CUDA-specific behavioral notes
        summary.append("")
        summary.append("CUDA KERNEL BEHAVIORAL NOTES:")
        has_logical_ops = any(instr.mnemonic in ['OR', 'AND', 'XOR'] for instr in analysis.instructions)
        has_bulk_ops = any('[I]' in instr.operands for instr in analysis.instructions)
        has_arithmetic = any(instr.mnemonic in ['ADD', 'SUB', 'SUBN'] for instr in analysis.instructions)
        
        if has_logical_ops:
            summary.append("  ⚠️  Contains OR/AND/XOR ops (CUDA sets VF=0, differs from standard CHIP-8)")
        if has_bulk_ops:
            summary.append("  📝 Contains bulk register ops (F55/F65 increment I register)")
        if has_arithmetic:
            summary.append("  🔢 Contains arithmetic ops (8-bit wraparound in CUDA)")
        
        # Sorting complexity assessment
        summary.append("")
        complexity_score = len(analysis.potential_sorting_instructions) + len(analysis.control_flow) * 2
        if complexity_score < 5:
            complexity = "SIMPLE"
        elif complexity_score < 15:
            complexity = "MODERATE" 
        else:
            complexity = "COMPLEX"
            
        summary.append(f"SORTING COMPLEXITY ASSESSMENT: {complexity}")
        summary.append(f"  Complexity score: {complexity_score}")
        summary.append(f"  Based on: {len(analysis.potential_sorting_instructions)} sorting instructions + {len(analysis.control_flow)} control flows")
        
        return "\n".join(summary)
    
    def analyze_sorting_rom(self, rom_path: Path, metadata_path: Path) -> SortingAnalysis:
        """Complete analysis of a sorting ROM"""
        
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
        
        # CRITICAL: Analyze authenticity
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

def generate_full_disassembly(analysis: SortingAnalysis, show_all: bool = False) -> str:
    """Generate full disassembly listing"""
    lines = []
    lines.append(f"=== FULL DISASSEMBLY: {analysis.filename} ===")
    lines.append("")
    
    # Show metadata summary
    sort_info = analysis.metadata.get('partial_sorting', {})
    lines.append(f"Sorting Achievement: {sort_info.get('sequence', [])} ({sort_info.get('direction', 'unknown')})")
    lines.append(f"Registers {sort_info.get('sequence_range', 'unknown')} sorted in {analysis.metadata.get('discovery_info', {}).get('sort_cycle', 'unknown')} cycles")
    lines.append("")
    
    # Disassembly
    lines.append("ADDRESS  OPCODE  MNEMONIC OPERANDS     DESCRIPTION")
    lines.append("-" * 70)
    
    for instr in analysis.instructions:
        # Mark potential sorting instructions
        marker = ">>> " if instr in analysis.potential_sorting_instructions else "    "
        
        line = f"{marker}${instr.address:03X}    ${instr.opcode:04X}   {instr.mnemonic:8} {instr.operands:12} {instr.description}"
        lines.append(line)
        
        # Show register effects for important instructions
        if not show_all and instr in analysis.potential_sorting_instructions:
            if instr.affects_registers:
                lines.append(f"          ^-- Modifies: {sorted(instr.affects_registers)}")
            if instr.reads_registers:
                lines.append(f"          ^-- Reads: {sorted(instr.reads_registers)}")
    
    return "\n".join(lines)

def main():
    parser = argparse.ArgumentParser(
        description="CHIP-8 Sorting ROM Decompiler - Analyze discovered sorting behavior",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python sorting_decompiler.py output/async_partial_sorting
  python sorting_decompiler.py output/async_partial_sorting --length 8
  python sorting_decompiler.py output/async_partial_sorting --rom LONGPARTIAL_B1234D01_V0-V7_L8_ASC_C9876_abcd1234.ch8
  python sorting_decompiler.py output/async_partial_sorting --summary-only
  python sorting_decompiler.py output/async_partial_sorting --full-disassembly
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
    
    # Initialize decompiler
    decompiler = CHIP8Decompiler()
    
    # Analyze each ROM
    analyses = []
    output_lines = []
    
    for i, (rom_file, json_file) in enumerate(filtered_pairs, 1):
        print(f"Analyzing ROM {i}/{len(filtered_pairs)}: {rom_file.name}")
        
        try:
            analysis = decompiler.analyze_sorting_rom(rom_file, json_file)
            analyses.append(analysis)
            
            # Add to output
            output_lines.append(analysis.analysis_summary)
            output_lines.append("")
            
            if args.full_disassembly and not args.summary_only:
                output_lines.append(generate_full_disassembly(analysis))
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
        # Length distribution
        length_counts = defaultdict(int)
        direction_counts = defaultdict(int)
        
        for analysis in analyses:
            sort_info = analysis.metadata.get('partial_sorting', {})
            length_counts[sort_info.get('length', 0)] += 1
            direction_counts[sort_info.get('direction', 'unknown')] += 1
        
        output_lines.append("")
        output_lines.append("Sorting length distribution:")
        for length in sorted(length_counts.keys()):
            output_lines.append(f"  {length}-element sorts: {length_counts[length]}")
        
        output_lines.append("")
        output_lines.append("Sorting direction distribution:")
        for direction, count in direction_counts.items():
            output_lines.append(f"  {direction}: {count}")
        
        # Authenticity analysis summary
        output_lines.append("")
        output_lines.append("AUTHENTICITY ANALYSIS:")
        genuine_count = sum(1 for a in analyses if getattr(a, 'authenticity', {}).get('is_genuine_sorting', False))
        coincidental_count = len(analyses) - genuine_count
        
        output_lines.append(f"  Genuine sorting algorithms: {genuine_count}")
        output_lines.append(f"  Coincidental consecutive values: {coincidental_count}")
        
        if genuine_count > 0:
            output_lines.append(f"  Genuine sorting rate: {genuine_count/len(analyses):.1%}")
        else:
            output_lines.append("  ⚠️  NO GENUINE SORTING FOUND - All appear to be coincidental!")
        
        # Classification breakdown
        classifications = {}
        for analysis in analyses:
            auth = getattr(analysis, 'authenticity', {})
            classification = auth.get('classification', 'UNKNOWN')
            if classification not in classifications:
                classifications[classification] = 0
            classifications[classification] += 1
        
        output_lines.append("")
        output_lines.append("Classification breakdown:")
        for classification, count in sorted(classifications.items()):
            output_lines.append(f"  {classification}: {count}")
        
        # Instruction complexity analysis
        avg_instructions = sum(len(a.instructions) for a in analyses) / len(analyses)
        avg_sorting_instructions = sum(len(a.potential_sorting_instructions) for a in analyses) / len(analyses)
        
        output_lines.append("")
        output_lines.append("Code complexity:")
        output_lines.append(f"  Average instructions per ROM: {avg_instructions:.1f}")
        output_lines.append(f"  Average sorting-related instructions: {avg_sorting_instructions:.1f}")
        
        # Most common sorting patterns
        sequences = []
        for analysis in analyses:
            seq = analysis.metadata.get('partial_sorting', {}).get('sequence', [])
            if seq:
                sequences.append(tuple(seq))
        
        if sequences:
            from collections import Counter
            common_sequences = Counter(sequences).most_common(10)
            output_lines.append("")
            output_lines.append("Most common sorting sequences:")
            for seq, count in common_sequences:
                # Mark if any of these sequences come from genuine sorting
                genuine_count_for_seq = 0
                for analysis in analyses:
                    if (tuple(analysis.metadata.get('partial_sorting', {}).get('sequence', [])) == seq and
                        getattr(analysis, 'authenticity', {}).get('is_genuine_sorting', False)):
                        genuine_count_for_seq += 1
                
                if genuine_count_for_seq > 0:
                    output_lines.append(f"  {list(seq)}: {count} occurrences ({genuine_count_for_seq} genuine)")
                else:
                    output_lines.append(f"  {list(seq)}: {count} occurrences (all coincidental)")
    
    # Output results
    output_text = "\n".join(output_lines)
    
    if args.output_file:
        with open(args.output_file, 'w', encoding='utf-8') as f:
            f.write(output_text)
        print(f"Analysis saved to: {args.output_file}")
    else:
        if not args.summary_only:
            print("\n" + "="*80)
            print("DETAILED ANALYSIS RESULTS:")
            print("="*80)
        print(output_text)
    
    print(f"\nAnalysis complete! Processed {len(analyses)} sorting ROMs.")
    
    # Show some interesting findings with authenticity focus
    if analyses:
        print("\nINTERESTING FINDINGS:")
        
        # Count genuine vs fake
        genuine_roms = [a for a in analyses if getattr(a, 'authenticity', {}).get('is_genuine_sorting', False)]
        fake_roms = [a for a in analyses if not getattr(a, 'authenticity', {}).get('is_genuine_sorting', False)]
        
        print(f"  Authenticity: {len(genuine_roms)} genuine, {len(fake_roms)} coincidental")
        
        if genuine_roms:
            # Find the shortest cycle count among genuine sorts
            shortest_cycle = min(a.metadata.get('discovery_info', {}).get('sort_cycle', float('inf')) for a in genuine_roms)
            fastest_rom = next(a for a in genuine_roms if a.metadata.get('discovery_info', {}).get('sort_cycle') == shortest_cycle)
            print(f"  Fastest GENUINE sorting: {fastest_rom.filename} in {shortest_cycle} cycles")
            
            # Find most complex genuine ROM
            most_complex_genuine = max(genuine_roms, key=lambda a: len(a.instructions))
            print(f"  Most complex genuine: {most_complex_genuine.filename} with {len(most_complex_genuine.instructions)} instructions")
        
        else:
            print(f"  ⚠️  CRITICAL: No genuine sorting algorithms found!")
            print(f"      All {len(analyses)} ROMs appear to be coincidental consecutive values")
            print(f"      The Babelscope may need adjustment to detect actual sorting vs. random consecutive placement")
        
        # Show most suspicious fake
        if fake_roms:
            # Find the one with most red flags
            most_suspicious = max(fake_roms, key=lambda a: len(getattr(a, 'authenticity', {}).get('red_flags', [])))
            red_flag_count = len(getattr(most_suspicious, 'authenticity', {}).get('red_flags', []))
            print(f"  Most suspicious fake: {most_suspicious.filename} with {red_flag_count} red flags")
    
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