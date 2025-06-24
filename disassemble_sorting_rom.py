#!/usr/bin/env python3
"""
CHIP-8 Sorting Algorithm Disassembler
Analyzes the 14 genuine sorting discoveries to reveal their assembly code
"""

import json
import os
import struct
from typing import List, Dict, Tuple

class CHIP8Disassembler:
    """CHIP-8 instruction disassembler"""
    
    def __init__(self):
        self.pc = 0x200  # Program counter starts at 0x200
        self.instructions = []
        
    def disassemble_instruction(self, opcode: int, address: int) -> str:
        """Disassemble a single CHIP-8 instruction"""
        # Extract nibbles
        n1 = (opcode & 0xF000) >> 12
        n2 = (opcode & 0x0F00) >> 8
        n3 = (opcode & 0x00F0) >> 4
        n4 = opcode & 0x000F
        
        nnn = opcode & 0x0FFF  # 12-bit address
        nn = opcode & 0x00FF   # 8-bit immediate
        x = n2                 # 4-bit register
        y = n3                 # 4-bit register
        
        # Instruction decoding
        if opcode == 0x00E0:
            return "CLS"
        elif opcode == 0x00EE:
            return "RET"
        elif n1 == 0x1:
            return f"JP   #{nnn:03X}"
        elif n1 == 0x2:
            return f"CALL #{nnn:03X}"
        elif n1 == 0x3:
            return f"SE   V{x:X}, #{nn:02X}"
        elif n1 == 0x4:
            return f"SNE  V{x:X}, #{nn:02X}"
        elif n1 == 0x5:
            return f"SE   V{x:X}, V{y:X}"
        elif n1 == 0x6:
            return f"LD   V{x:X}, #{nn:02X}"
        elif n1 == 0x7:
            return f"ADD  V{x:X}, #{nn:02X}"
        elif n1 == 0x8:
            if n4 == 0x0:
                return f"LD   V{x:X}, V{y:X}"
            elif n4 == 0x1:
                return f"OR   V{x:X}, V{y:X}"
            elif n4 == 0x2:
                return f"AND  V{x:X}, V{y:X}"
            elif n4 == 0x3:
                return f"XOR  V{x:X}, V{y:X}"
            elif n4 == 0x4:
                return f"ADD  V{x:X}, V{y:X}"
            elif n4 == 0x5:
                return f"SUB  V{x:X}, V{y:X}"
            elif n4 == 0x6:
                return f"SHR  V{x:X}"
            elif n4 == 0x7:
                return f"SUBN V{x:X}, V{y:X}"
            elif n4 == 0xE:
                return f"SHL  V{x:X}"
        elif n1 == 0x9:
            return f"SNE  V{x:X}, V{y:X}"
        elif n1 == 0xA:
            return f"LD   I, #{nnn:03X}"
        elif n1 == 0xB:
            return f"JP   V0, #{nnn:03X}"
        elif n1 == 0xC:
            return f"RND  V{x:X}, #{nn:02X}"
        elif n1 == 0xD:
            return f"DRW  V{x:X}, V{y:X}, #{n4:X}"
        elif n1 == 0xE:
            if nn == 0x9E:
                return f"SKP  V{x:X}"
            elif nn == 0xA1:
                return f"SKNP V{x:X}"
        elif n1 == 0xF:
            if nn == 0x07:
                return f"LD   V{x:X}, DT"
            elif nn == 0x0A:
                return f"LD   V{x:X}, K"
            elif nn == 0x15:
                return f"LD   DT, V{x:X}"
            elif nn == 0x18:
                return f"LD   ST, V{x:X}"
            elif nn == 0x1E:
                return f"ADD  I, V{x:X}"
            elif nn == 0x29:
                return f"LD   F, V{x:X}"
            elif nn == 0x33:
                return f"LD   B, V{x:X}"
            elif nn == 0x55:
                return f"LD   [I], V{x:X}"  # Store V0-VX to memory
            elif nn == 0x65:
                return f"LD   V{x:X}, [I]"  # Load V0-VX from memory
        
        return f"DATA #{opcode:04X}"  # Unknown instruction
    
    def disassemble_rom(self, rom_data: bytes) -> List[Tuple[int, int, str]]:
        """Disassemble entire ROM"""
        instructions = []
        address = 0x200
        
        # Process ROM in 2-byte chunks
        for i in range(0, len(rom_data), 2):
            if i + 1 < len(rom_data):
                # Big-endian 16-bit instruction
                opcode = (rom_data[i] << 8) | rom_data[i + 1]
                disasm = self.disassemble_instruction(opcode, address)
                instructions.append((address, opcode, disasm))
                address += 2
        
        return instructions

def load_genuine_discoveries(discoveries_file: str) -> List[Dict]:
    """Load and filter the 14 genuine sorting discoveries"""
    with open(discoveries_file, 'r') as f:
        discoveries = json.load(f)
    
    # Filter for genuine discoveries (array_reads > 0)
    genuine = [d for d in discoveries if d['array_reads'] > 0]
    
    # Sort by genuineness score (calculated here)
    def calculate_score(d):
        score = d['array_reads'] * 5
        score += min(d['comparisons'] / 10, 20)
        score += min(d['swaps'], 15)
        score += min(len(set(d['final_array'])) * 2, 15)
        if len(set(d['final_array'])) == len(d['final_array']):  # All unique
            score += 10
        return score
    
    genuine.sort(key=calculate_score, reverse=True)
    return genuine

def reconstruct_rom_from_filename(filename: str) -> bytes:
    """
    Reconstruct ROM data from filename hash (placeholder)
    In reality, you'd need the actual ROM files or regeneration logic
    """
    # Extract hash from filename (last part before .ch8)
    parts = filename.split('_')
    if len(parts) >= 5:
        hash_part = parts[-1].replace('.ch8', '')
        print(f"ROM hash: {hash_part}")
    
    # For now, return dummy ROM data
    # You'd need to implement actual ROM reconstruction here
    return b'\x00' * 512  # Placeholder

def analyze_sorting_behavior(instructions: List[Tuple[int, int, str]], discovery: Dict) -> Dict:
    """Analyze the disassembled code for sorting-related patterns"""
    analysis = {
        'memory_operations': [],
        'comparison_patterns': [],
        'loop_structures': [],
        'register_usage': set(),
        'potential_algorithm': 'Unknown'
    }
    
    for addr, opcode, disasm in instructions:
        # Track memory operations
        if 'LD   [I]' in disasm or 'LD   V' in disasm and '[I]' in disasm:
            analysis['memory_operations'].append((addr, disasm))
        
        # Track comparisons
        if disasm.startswith(('SE ', 'SNE ', 'SKP ', 'SKNP ')):
            analysis['comparison_patterns'].append((addr, disasm))
        
        # Track register usage
        import re
        regs = re.findall(r'V([0-9A-F])', disasm)
        analysis['register_usage'].update(regs)
        
        # Detect potential loops (jumps backward)
        if disasm.startswith('JP ') and '#' in disasm:
            target = int(disasm.split('#')[1], 16)
            if target < addr:
                analysis['loop_structures'].append((addr, target, disasm))
    
    # Simple algorithm detection heuristics
    if discovery['swaps'] > 100:
        analysis['potential_algorithm'] = 'Bubble Sort (high swap count)'
    elif discovery['comparisons'] > discovery['swaps'] * 10:
        analysis['potential_algorithm'] = 'Selection Sort (comparison heavy)'
    elif discovery['swaps'] < 10 and discovery['comparisons'] > 50:
        analysis['potential_algorithm'] = 'Insertion Sort (few swaps)'
    
    return analysis

def main():
    """Main disassembly and analysis function"""
    discoveries_file = 'discoveries.json'
    
    print("🔍 CHIP-8 Sorting Algorithm Disassembler")
    print("=" * 50)
    
    if not os.path.exists(discoveries_file):
        print(f"❌ Error: {discoveries_file} not found!")
        return
    
    # Load genuine discoveries
    genuine_discoveries = load_genuine_discoveries(discoveries_file)
    print(f"📊 Found {len(genuine_discoveries)} genuine sorting discoveries")
    print()
    
    disassembler = CHIP8Disassembler()
    
    for i, discovery in enumerate(genuine_discoveries, 1):
        print(f"🏆 DISCOVERY #{i}: Batch {discovery['batch']}, Instance {discovery['instance_id']}")
        print(f"📋 Operations: {discovery['array_reads']} reads, {discovery['array_writes']} writes, "
              f"{discovery['comparisons']} comparisons, {discovery['swaps']} swaps")
        print(f"📊 Final array: {discovery['final_array']}")
        print(f"📁 ROM file: {discovery['rom_filename']}")
        print()
        
        # Reconstruct ROM (placeholder - you'd need actual ROM data)
        rom_data = reconstruct_rom_from_filename(discovery['rom_filename'])
        
        # Disassemble
        instructions = disassembler.disassemble_rom(rom_data)
        
        print("🔧 DISASSEMBLY:")
        print("-" * 30)
        for addr, opcode, disasm in instructions[:20]:  # Show first 20 instructions
            print(f"{addr:03X}: {opcode:04X}  {disasm}")
        
        if len(instructions) > 20:
            print(f"... ({len(instructions) - 20} more instructions)")
        print()
        
        # Analyze sorting behavior
        analysis = analyze_sorting_behavior(instructions, discovery)
        
        print("🧠 BEHAVIOR ANALYSIS:")
        print(f"• Memory operations: {len(analysis['memory_operations'])}")
        print(f"• Comparison patterns: {len(analysis['comparison_patterns'])}")
        print(f"• Loop structures: {len(analysis['loop_structures'])}")
        print(f"• Registers used: {sorted(analysis['register_usage'])}")
        print(f"• Likely algorithm: {analysis['potential_algorithm']}")
        print()
        print("=" * 50)
        print()

if __name__ == "__main__":
    main()