#!/usr/bin/env python3
"""
CHIP-8 ROM Disassembler for Babelscope Discoveries
Disassembles discovered ROMs to show the actual machine code that created sorting algorithms
"""

import os
import json
import argparse
from typing import List, Dict, Tuple, Optional

class CHIP8Disassembler:
    """
    Complete CHIP-8 disassembler with analysis for Babelscope discoveries
    """
    
    def __init__(self):
        self.font_data = [
            0xF0, 0x90, 0x90, 0x90, 0xF0,  # 0
            0x20, 0x60, 0x20, 0x20, 0x70,  # 1
            0xF0, 0x10, 0xF0, 0x80, 0xF0,  # 2
            0xF0, 0x10, 0xF0, 0x10, 0xF0,  # 3
            0x90, 0x90, 0xF0, 0x10, 0x10,  # 4
            0xF0, 0x80, 0xF0, 0x10, 0xF0,  # 5
            0xF0, 0x80, 0xF0, 0x90, 0xF0,  # 6
            0xF0, 0x10, 0x20, 0x40, 0x40,  # 7
            0xF0, 0x90, 0xF0, 0x90, 0xF0,  # 8
            0xF0, 0x90, 0xF0, 0x10, 0xF0,  # 9
            0xF0, 0x90, 0xF0, 0x90, 0x90,  # A
            0xE0, 0x90, 0xE0, 0x90, 0xE0,  # B
            0xF0, 0x80, 0x80, 0x80, 0xF0,  # C
            0xE0, 0x90, 0x90, 0x90, 0xE0,  # D
            0xF0, 0x80, 0xF0, 0x80, 0xF0,  # E
            0xF0, 0x80, 0xF0, 0x80, 0x80   # F
        ]
    
    def disassemble_instruction(self, instruction: int, address: int) -> Tuple[str, str, str]:
        """
        Disassemble a single CHIP-8 instruction
        Returns: (mnemonic, operands, description)
        """
        
        # Extract components
        opcode = (instruction & 0xF000) >> 12
        x = (instruction & 0x0F00) >> 8
        y = (instruction & 0x00F0) >> 4
        n = instruction & 0x000F
        kk = instruction & 0x00FF
        nnn = instruction & 0x0FFF
        
        # Disassemble based on opcode
        if instruction == 0x00E0:
            return "CLS", "", "Clear display"
        elif instruction == 0x00EE:
            return "RET", "", "Return from subroutine"
        elif opcode == 0x0:
            return "SYS", f"${nnn:03X}", f"System call to {nnn:03X}"
        elif opcode == 0x1:
            return "JP", f"${nnn:03X}", f"Jump to {nnn:03X}"
        elif opcode == 0x2:
            return "CALL", f"${nnn:03X}", f"Call subroutine at {nnn:03X}"
        elif opcode == 0x3:
            return "SE", f"V{x:X}, #{kk:02X}", f"Skip if V{x:X} == {kk}"
        elif opcode == 0x4:
            return "SNE", f"V{x:X}, #{kk:02X}", f"Skip if V{x:X} != {kk}"
        elif opcode == 0x5:
            if n == 0:
                return "SE", f"V{x:X}, V{y:X}", f"Skip if V{x:X} == V{y:X}"
            else:
                return "UNKNOWN", f"${instruction:04X}", "Unknown 5xxx instruction"
        elif opcode == 0x6:
            return "LD", f"V{x:X}, #{kk:02X}", f"Load {kk} into V{x:X}"
        elif opcode == 0x7:
            return "ADD", f"V{x:X}, #{kk:02X}", f"Add {kk} to V{x:X}"
        elif opcode == 0x8:
            return self._disassemble_8xxx(x, y, n)
        elif opcode == 0x9:
            if n == 0:
                return "SNE", f"V{x:X}, V{y:X}", f"Skip if V{x:X} != V{y:X}"
            else:
                return "UNKNOWN", f"${instruction:04X}", "Unknown 9xxx instruction"
        elif opcode == 0xA:
            return "LD", f"I, ${nnn:03X}", f"Load {nnn:03X} into I"
        elif opcode == 0xB:
            return "JP", f"V0, ${nnn:03X}", f"Jump to V0 + {nnn:03X}"
        elif opcode == 0xC:
            return "RND", f"V{x:X}, #{kk:02X}", f"V{x:X} = random & {kk:02X}"
        elif opcode == 0xD:
            return "DRW", f"V{x:X}, V{y:X}, #{n:X}", f"Draw {n}-byte sprite at V{x:X}, V{y:X}"
        elif opcode == 0xE:
            if kk == 0x9E:
                return "SKP", f"V{x:X}", f"Skip if key V{x:X} pressed"
            elif kk == 0xA1:
                return "SKNP", f"V{x:X}", f"Skip if key V{x:X} not pressed"
            else:
                return "UNKNOWN", f"${instruction:04X}", "Unknown Exxx instruction"
        elif opcode == 0xF:
            return self._disassemble_fxxx(x, kk)
        else:
            return "UNKNOWN", f"${instruction:04X}", "Unknown instruction"
    
    def _disassemble_8xxx(self, x: int, y: int, n: int) -> Tuple[str, str, str]:
        """Disassemble 8xxx register operations"""
        if n == 0x0:
            return "LD", f"V{x:X}, V{y:X}", f"V{x:X} = V{y:X}"
        elif n == 0x1:
            return "OR", f"V{x:X}, V{y:X}", f"V{x:X} |= V{y:X}"
        elif n == 0x2:
            return "AND", f"V{x:X}, V{y:X}", f"V{x:X} &= V{y:X}"
        elif n == 0x3:
            return "XOR", f"V{x:X}, V{y:X}", f"V{x:X} ^= V{y:X}"
        elif n == 0x4:
            return "ADD", f"V{x:X}, V{y:X}", f"V{x:X} += V{y:X}, VF = carry"
        elif n == 0x5:
            return "SUB", f"V{x:X}, V{y:X}", f"V{x:X} -= V{y:X}, VF = !borrow"
        elif n == 0x6:
            return "SHR", f"V{x:X}", f"V{x:X} >>= 1, VF = LSB"
        elif n == 0x7:
            return "SUBN", f"V{x:X}, V{y:X}", f"V{x:X} = V{y:X} - V{x:X}, VF = !borrow"
        elif n == 0xE:
            return "SHL", f"V{x:X}", f"V{x:X} <<= 1, VF = MSB"
        else:
            return "UNKNOWN", f"8{x:X}{y:X}{n:X}", f"Unknown 8xxx instruction"
    
    def _disassemble_fxxx(self, x: int, kk: int) -> Tuple[str, str, str]:
        """Disassemble Fxxx timer and memory operations"""
        if kk == 0x07:
            return "LD", f"V{x:X}, DT", f"V{x:X} = delay timer"
        elif kk == 0x0A:
            return "LD", f"V{x:X}, K", f"Wait for key press, store in V{x:X}"
        elif kk == 0x15:
            return "LD", f"DT, V{x:X}", f"Delay timer = V{x:X}"
        elif kk == 0x18:
            return "LD", f"ST, V{x:X}", f"Sound timer = V{x:X}"
        elif kk == 0x1E:
            return "ADD", f"I, V{x:X}", f"I += V{x:X}"
        elif kk == 0x29:
            return "LD", f"F, V{x:X}", f"I = sprite address for digit V{x:X}"
        elif kk == 0x33:
            return "LD", f"B, V{x:X}", f"Store BCD of V{x:X} at I, I+1, I+2"
        elif kk == 0x55:
            return "LD", f"[I], V{x:X}", f"Store V0-V{x:X} at I"
        elif kk == 0x65:
            return "LD", f"V{x:X}, [I]", f"Load V0-V{x:X} from I"
        else:
            return "UNKNOWN", f"F{x:X}{kk:02X}", f"Unknown Fxxx instruction"
    
    def find_sort_array_references(self, disassembly: List[Tuple[int, int, str, str, str]]) -> List[Tuple[int, str]]:
        """Find instructions that reference the sort array at 0x300-0x307"""
        references = []
        
        for address, instruction, mnemonic, operands, description in disassembly:
            # Look for memory operations with I register
            if mnemonic in ["LD"] and ("[I]" in operands or "I," in operands):
                references.append((address, f"{mnemonic} {operands} ; {description}"))
            
            # Look for loads to I register that might be setting up for sort array access
            elif mnemonic == "LD" and "I," in operands:
                # Extract the address being loaded
                if "$" in operands:
                    addr_str = operands.split("$")[1]
                    try:
                        addr = int(addr_str, 16)
                        if 0x300 <= addr <= 0x320:  # Near sort array
                            references.append((address, f"{mnemonic} {operands} ; SORT ARRAY SETUP"))
                    except ValueError:
                        pass
            
            # Look for arithmetic that might compute sort array addresses
            elif mnemonic == "ADD" and "I," in operands:
                references.append((address, f"{mnemonic} {operands} ; I ADDRESS MANIPULATION"))
        
        return references
    
    def analyze_control_flow(self, disassembly: List[Tuple[int, int, str, str, str]]) -> Dict[str, List[int]]:
        """Analyze control flow patterns (loops, branches, calls)"""
        analysis = {
            'jumps': [],
            'calls': [],
            'branches': [],
            'loops': []
        }
        
        for address, instruction, mnemonic, operands, description in disassembly:
            if mnemonic == "JP":
                target = self._extract_address(operands)
                if target:
                    analysis['jumps'].append((address, target))
                    # Check for backward jump (potential loop)
                    if target <= address:
                        analysis['loops'].append((address, target))
            
            elif mnemonic == "CALL":
                target = self._extract_address(operands)
                if target:
                    analysis['calls'].append((address, target))
            
            elif mnemonic in ["SE", "SNE", "SKP", "SKNP"]:
                analysis['branches'].append(address)
        
        return analysis
    
    def _extract_address(self, operands: str) -> Optional[int]:
        """Extract address from operands string"""
        if "$" in operands:
            addr_str = operands.split("$")[1].split(",")[0].strip()
            try:
                return int(addr_str, 16)
            except ValueError:
                pass
        return None
    
    def disassemble_rom(self, rom_data: bytes, start_address: int = 0x200) -> List[Tuple[int, int, str, str, str]]:
        """
        Disassemble entire ROM
        Returns list of (address, instruction, mnemonic, operands, description)
        """
        disassembly = []
        address = start_address
        
        # Process instructions in pairs (CHIP-8 instructions are 2 bytes)
        for i in range(0, len(rom_data) - 1, 2):
            if i + 1 < len(rom_data):
                # Combine two bytes into 16-bit instruction (big-endian)
                instruction = (rom_data[i] << 8) | rom_data[i + 1]
                
                # Skip if this looks like data (all zeros or font data)
                if instruction == 0x0000:
                    address += 2
                    continue
                
                mnemonic, operands, description = self.disassemble_instruction(instruction, address)
                disassembly.append((address, instruction, mnemonic, operands, description))
                
                address += 2
        
        return disassembly
    
    def generate_analysis_report(self, rom_file: str, discovery_info: Dict = None) -> str:
        """Generate comprehensive analysis report for a ROM"""
        
        # Load ROM
        try:
            with open(rom_file, 'rb') as f:
                rom_data = f.read()
        except FileNotFoundError:
            return f"❌ ROM file not found: {rom_file}"
        
        # Disassemble
        disassembly = self.disassemble_rom(rom_data)
        
        if not disassembly:
            return f"❌ No valid instructions found in {rom_file}"
        
        # Analyze
        sort_refs = self.find_sort_array_references(disassembly)
        control_flow = self.analyze_control_flow(disassembly)
        
        # Generate report
        report = f"""
CHIP-8 ROM DISASSEMBLY ANALYSIS
{'='*60}
ROM File: {rom_file}
ROM Size: {len(rom_data)} bytes
Instructions Found: {len(disassembly)}

"""
        
        # Add discovery info if provided
        if discovery_info:
            report += f"""DISCOVERY INFORMATION:
Batch: {discovery_info.get('batch', 'Unknown')}
Instance: {discovery_info.get('instance_id', 'Unknown')}
Direction: {discovery_info.get('direction', 'Unknown')}
Final Array: {discovery_info.get('final_array', [])}
Operations: R:{discovery_info.get('array_reads', 0)} W:{discovery_info.get('array_writes', 0)} C:{discovery_info.get('comparisons', 0)} S:{discovery_info.get('swaps', 0)}

"""
        
        # Sort array analysis
        report += f"""SORT ARRAY ANALYSIS (0x300-0x307):
References Found: {len(sort_refs)}
"""
        
        if sort_refs:
            report += "Sort Array References:\n"
            for addr, instruction in sort_refs:
                report += f"  ${addr:03X}: {instruction}\n"
        else:
            report += "WARNING: No direct sort array references found\n"
        
        report += f"""
CONTROL FLOW ANALYSIS:
Jumps: {len(control_flow['jumps'])}
Calls: {len(control_flow['calls'])}
Branches: {len(control_flow['branches'])}
Loops: {len(control_flow['loops'])}
"""
        
        if control_flow['loops']:
            report += "Loop Structures:\n"
            for start_addr, target_addr in control_flow['loops']:
                report += f"  ${start_addr:03X} -> ${target_addr:03X} (backward jump)\n"
        
        # Full disassembly
        report += f"""
COMPLETE DISASSEMBLY:
{'='*40}
Address  Opcode  Mnemonic Operands           Description
{'='*60}
"""
        
        for address, instruction, mnemonic, operands, description in disassembly:
            report += f"${address:03X}    ${instruction:04X}   {mnemonic:<8} {operands:<15} ; {description}\n"
        
        # Look for interesting patterns
        report += f"""
ALGORITHMIC ANALYSIS:
{'='*30}
"""
        
        # Count instruction types
        instruction_counts = {}
        for _, _, mnemonic, _, _ in disassembly:
            instruction_counts[mnemonic] = instruction_counts.get(mnemonic, 0) + 1
        
        report += "Instruction Distribution:\n"
        for mnemonic, count in sorted(instruction_counts.items(), key=lambda x: x[1], reverse=True):
            percentage = count / len(disassembly) * 100
            report += f"  {mnemonic:<8}: {count:3d} ({percentage:4.1f}%)\n"
        
        # Memory operations analysis
        memory_ops = [mnem for mnem in instruction_counts.keys() if mnem in ['LD', 'ADD'] and instruction_counts[mnem] > 0]
        if memory_ops:
            total_memory_ops = sum(instruction_counts[op] for op in memory_ops)
            report += f"\nMemory Operations: {total_memory_ops} ({total_memory_ops/len(disassembly)*100:.1f}%)\n"
        
        # Arithmetic operations
        arith_ops = [mnem for mnem in instruction_counts.keys() if mnem in ['ADD', 'SUB', 'SUBN', 'AND', 'OR', 'XOR']]
        if arith_ops:
            total_arith = sum(instruction_counts[op] for op in arith_ops)
            report += f"Arithmetic Operations: {total_arith} ({total_arith/len(disassembly)*100:.1f}%)\n"
        
        return report
    
    def batch_analyze_discoveries(self, discoveries_file: str, rom_directory: str, output_dir: str = "disassembly_analysis"):
        """Analyze all discovered ROMs and generate reports"""
        
        # Load discoveries
        try:
            with open(discoveries_file, 'r') as f:
                discoveries = json.load(f)
        except FileNotFoundError:
            print(f"❌ Discoveries file not found: {discoveries_file}")
            return
        
        os.makedirs(output_dir, exist_ok=True)
        
        print(f"Analyzing {len(discoveries)} discovered ROMs...")
        
        analysis_summary = []
        
        for i, discovery in enumerate(discoveries):
            rom_filename = discovery['rom_filename']
            rom_path = os.path.join(rom_directory, rom_filename)
            
            if not os.path.exists(rom_path):
                print(f"WARNING: ROM not found: {rom_path}")
                continue
            
            print(f"   Analyzing {i+1}/{len(discoveries)}: {rom_filename}")
            
            # Generate analysis report
            report = self.generate_analysis_report(rom_path, discovery)
            
            # Save individual report
            output_file = os.path.join(output_dir, f"{rom_filename}_analysis.txt")
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(report)
            
            # Collect summary info
            summary_info = {
                'rom_filename': rom_filename,
                'batch': discovery['batch'],
                'instance_id': discovery['instance_id'],
                'final_array': discovery['final_array'],
                'array_reads': discovery['array_reads'],
                'analysis_file': output_file
            }
            analysis_summary.append(summary_info)
        
        # Save summary
        summary_file = os.path.join(output_dir, "analysis_summary.json")
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(analysis_summary, f, indent=2)
        
        print(f"\nAnalysis complete!")
        print(f"Reports saved to: {output_dir}/")
        print(f"Summary: {summary_file}")


def main():
    """Main disassembly routine"""
    parser = argparse.ArgumentParser(description='CHIP-8 ROM Disassembler for Babelscope Discoveries')
    parser.add_argument('--rom', help='Single ROM file to analyze')
    parser.add_argument('--discoveries', help='JSON file containing all discoveries')
    parser.add_argument('--rom-dir', default='output/sorting', help='Directory containing ROM files')
    parser.add_argument('--output-dir', default='disassembly_analysis', help='Output directory for analysis')
    
    args = parser.parse_args()
    
    disassembler = CHIP8Disassembler()
    
    if args.rom:
        # Analyze single ROM
        print(f"Analyzing single ROM: {args.rom}")
        report = disassembler.generate_analysis_report(args.rom)
        print(report)
        
        # Save report
        output_file = f"{args.rom}_analysis.txt"
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(report)
        print(f"\nAnalysis saved to: {output_file}")
    
    elif args.discoveries:
        # Batch analyze all discoveries
        disassembler.batch_analyze_discoveries(args.discoveries, args.rom_dir, args.output_dir)
    
    else:
        print("ERROR: Please specify either --rom for single analysis or --discoveries for batch analysis")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())