#!/usr/bin/env python3
"""
Babelscope: Massively Parallel CHIP-8 Emulator
Using Python/CuPy for computational space exploration

A pure CuPy implementation without raw CUDA kernels for better reliability.
Based on Brian Benchoff's Finite Atari Machine approach.
"""

import cupy as cp
import numpy as np
import time
import hashlib
from pathlib import Path
from typing import Tuple, Dict, List
import json

# =============================================================================
# CHIP-8 Constants
# =============================================================================

MEMORY_SIZE = 4096
DISPLAY_WIDTH = 64
DISPLAY_HEIGHT = 32
DISPLAY_SIZE = DISPLAY_WIDTH * DISPLAY_HEIGHT
REGISTER_COUNT = 16
STACK_SIZE = 16
ROM_SIZE = 3584  # 4KB - 512 bytes (interpreter space)  
MAX_CYCLES = 1000  # Reduced for batch processing

# Batch configuration - Conservative for stability
BATCH_SIZE = 1024 * 80  # 80k ROMs per batch
STATUS_EVERY = 5        # Batches between status updates

# Analysis thresholds (relaxed for random data discovery)
MIN_DISPLAY_WRITES = 1      # Must actually draw something
MIN_CYCLES = 10             # Must execute reasonable amount
MIN_UNIQUE_OPCODES = 3      # Must use varied instruction set
MIN_MEMORY_WRITES = 0       # Don't require memory writes
MIN_PIXELS_SET = 1          # Must have visible output
MIN_SCORE = 0.05            # Very low composite threshold

# Output configuration
OUTPUT_DIR = Path("babelscope_roms")
OUTPUT_DIR.mkdir(exist_ok=True)

# =============================================================================
# CHIP-8 Font Set (loaded at 0x50)
# =============================================================================

CHIP8_FONTSET = cp.array([
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
], dtype=cp.uint8)

# =============================================================================
# Pure CuPy CHIP-8 Emulator Functions
# =============================================================================

def create_batch_roms(batch_size: int) -> cp.ndarray:
    """Generate a batch of random CHIP-8 ROMs on GPU"""
    # Check available GPU memory
    free_mem, total_mem = cp.cuda.runtime.memGetInfo()
    free_mem_mb = free_mem // (1024 * 1024)
    
    # Estimate memory per ROM (conservative)
    memory_per_rom = ROM_SIZE + MEMORY_SIZE + DISPLAY_SIZE + 100  # ROM + emulator state + overhead
    estimated_memory_mb = (batch_size * memory_per_rom) // (1024 * 1024)
    
    if estimated_memory_mb > free_mem_mb * 0.6:  # Use max 60% of free memory
        max_batch_size = int((free_mem_mb * 0.6 * 1024 * 1024) // memory_per_rom)
        print(f"Warning: Reducing batch size from {batch_size:,} to {max_batch_size:,} to fit in GPU memory")
        batch_size = max_batch_size
    
    return cp.random.randint(0, 256, size=(batch_size, ROM_SIZE), dtype=cp.uint8)

def init_chip8_batch(roms: cp.ndarray) -> Dict[str, cp.ndarray]:
    """Initialize CHIP-8 systems for a batch of ROMs"""
    batch_size = roms.shape[0]
    
    # System state for entire batch
    memory = cp.zeros((batch_size, MEMORY_SIZE), dtype=cp.uint8)
    display = cp.zeros((batch_size, DISPLAY_SIZE), dtype=cp.uint8)
    V = cp.zeros((batch_size, 16), dtype=cp.uint8)  # Registers V0-VF
    I = cp.zeros(batch_size, dtype=cp.uint16)       # Index register
    pc = cp.full(batch_size, 0x200, dtype=cp.uint16)  # Program counter
    sp = cp.zeros(batch_size, dtype=cp.uint8)       # Stack pointer
    stack = cp.zeros((batch_size, 16), dtype=cp.uint16)
    delay_timer = cp.zeros(batch_size, dtype=cp.uint8)
    sound_timer = cp.zeros(batch_size, dtype=cp.uint8)
    
    # Load font set into memory for all ROMs
    for i in range(80):
        memory[:, 0x50 + i] = CHIP8_FONTSET[i]
    
    # Load ROM data into memory
    memory[:, 0x200:0x200+ROM_SIZE] = roms
    
    # Analysis metrics
    cycles_executed = cp.zeros(batch_size, dtype=cp.uint32)
    display_writes = cp.zeros(batch_size, dtype=cp.uint32)
    memory_writes = cp.zeros(batch_size, dtype=cp.uint32)
    unique_opcodes = cp.zeros((batch_size, 256), dtype=cp.bool_)
    
    return {
        'memory': memory,
        'display': display,
        'V': V,
        'I': I,
        'pc': pc,
        'sp': sp,
        'stack': stack,
        'delay_timer': delay_timer,
        'sound_timer': sound_timer,
        'cycles_executed': cycles_executed,
        'display_writes': display_writes,
        'memory_writes': memory_writes,
        'unique_opcodes': unique_opcodes,
        'active': cp.ones(batch_size, dtype=cp.bool_)  # Which ROMs are still running
    }

def execute_instruction_batch(state: Dict[str, cp.ndarray]):
    """Execute one instruction for all active ROMs in the batch"""
    active = state['active']
    
    # Only process active ROMs
    if not cp.any(active):
        return
    
    # Fetch instructions for active ROMs
    pc_vals = state['pc'][active]
    memory = state['memory'][active]
    
    # Bounds check
    valid_pc = (pc_vals < MEMORY_SIZE - 1)
    if not cp.any(valid_pc):
        state['active'][active] = False
        return
    
    # Get instruction bytes
    instr_high = memory[cp.arange(len(pc_vals)), pc_vals]
    instr_low = memory[cp.arange(len(pc_vals)), pc_vals + 1]
    instructions = (instr_high.astype(cp.uint16) << 8) | instr_low.astype(cp.uint16)
    
    # Update PC for active ROMs
    state['pc'][active] += 2
    state['cycles_executed'][active] += 1
    
    # Track opcode usage
    opcode_families = instructions >> 8
    for i, family in enumerate(opcode_families):
        if valid_pc[i]:
            idx = cp.where(active)[0][i]
            state['unique_opcodes'][idx, family] = True
    
    # Decode instruction parts
    nnn = instructions & 0x0FFF
    nn = instructions & 0x00FF
    n = instructions & 0x000F
    x = (instructions & 0x0F00) >> 8
    y = (instructions & 0x00F0) >> 4
    
    # Execute instructions by family
    execute_0xxx_family(state, active, instructions, nn)
    execute_1xxx_family(state, active, instructions, nnn)
    execute_6xxx_family(state, active, instructions, x, nn)
    execute_7xxx_family(state, active, instructions, x, nn)
    execute_axxx_family(state, active, instructions, nnn)
    execute_dxxx_family(state, active, instructions, x, y, n)
    
    # Update timers for active ROMs
    state['delay_timer'][active] = cp.maximum(0, state['delay_timer'][active] - 1)
    state['sound_timer'][active] = cp.maximum(0, state['sound_timer'][active] - 1)

def execute_0xxx_family(state, active, instructions, nn):
    """Execute 0xxx family instructions"""
    # 00E0 - CLS (Clear display)
    cls_mask = (instructions == 0x00E0)
    if cp.any(cls_mask):
        active_indices = cp.where(active)[0]
        cls_indices = active_indices[cls_mask]
        state['display'][cls_indices, :] = 0
        state['display_writes'][cls_indices] += 1

def execute_1xxx_family(state, active, instructions, nnn):
    """Execute 1xxx family instructions"""
    # 1NNN - JP addr (Jump to location nnn)
    jp_mask = ((instructions & 0xF000) == 0x1000)
    if cp.any(jp_mask):
        active_indices = cp.where(active)[0]
        jp_indices = active_indices[jp_mask]
        nnn_vals = cp.asnumpy(nnn[jp_mask])
        jp_indices_cpu = cp.asnumpy(jp_indices)
        
        for i in range(len(jp_indices_cpu)):
            state['pc'][jp_indices_cpu[i]] = int(nnn_vals[i])

def execute_6xxx_family(state, active, instructions, x, nn):
    """Execute 6xxx family instructions"""
    # 6XNN - LD Vx, byte (Set Vx = nn)
    ld_mask = ((instructions & 0xF000) == 0x6000)
    if cp.any(ld_mask):
        active_indices = cp.where(active)[0]
        ld_indices = active_indices[ld_mask]
        x_vals = cp.asnumpy(x[ld_mask])
        nn_vals = cp.asnumpy(nn[ld_mask])
        ld_indices_cpu = cp.asnumpy(ld_indices)
        
        # Use advanced indexing properly
        for i in range(len(ld_indices_cpu)):
            x_reg = int(x_vals[i])
            if x_reg < 16:  # Bounds check
                state['V'][ld_indices_cpu[i], x_reg] = int(nn_vals[i])

def execute_7xxx_family(state, active, instructions, x, nn):
    """Execute 7xxx family instructions"""
    # 7XNN - ADD Vx, byte (Set Vx = Vx + nn)
    add_mask = ((instructions & 0xF000) == 0x7000)
    if cp.any(add_mask):
        active_indices = cp.where(active)[0]
        add_indices = active_indices[add_mask]
        x_vals = cp.asnumpy(x[add_mask])
        nn_vals = cp.asnumpy(nn[add_mask])
        add_indices_cpu = cp.asnumpy(add_indices)
        
        # Use advanced indexing properly
        for i in range(len(add_indices_cpu)):
            x_reg = int(x_vals[i])
            if x_reg < 16:  # Bounds check
                old_val = int(state['V'][add_indices_cpu[i], x_reg])
                state['V'][add_indices_cpu[i], x_reg] = (old_val + int(nn_vals[i])) & 0xFF

def execute_axxx_family(state, active, instructions, nnn):
    """Execute Axxx family instructions"""
    # ANNN - LD I, addr (Set I = nnn)
    ldi_mask = ((instructions & 0xF000) == 0xA000)
    if cp.any(ldi_mask):
        active_indices = cp.where(active)[0]
        ldi_indices = active_indices[ldi_mask]
        nnn_vals = cp.asnumpy(nnn[ldi_mask])
        ldi_indices_cpu = cp.asnumpy(ldi_indices)
        
        for i in range(len(ldi_indices_cpu)):
            state['I'][ldi_indices_cpu[i]] = int(nnn_vals[i])

def execute_dxxx_family(state, active, instructions, x, y, n):
    """Execute Dxxx family instructions - Draw sprite"""
    # DXYN - DRW Vx, Vy, nibble
    drw_mask = ((instructions & 0xF000) == 0xD000)
    if not cp.any(drw_mask):
        return
    
    active_indices = cp.where(active)[0]
    drw_indices = active_indices[drw_mask]
    
    if len(drw_indices) == 0:
        return
    
    # Get sprite parameters and convert to CPU arrays for processing
    x_vals = cp.asnumpy(x[drw_mask])
    y_vals = cp.asnumpy(y[drw_mask])
    height_vals = cp.asnumpy(n[drw_mask])
    drw_indices_cpu = cp.asnumpy(drw_indices)
    
    # Process each drawing operation individually to avoid indexing issues
    for i in range(len(drw_indices_cpu)):
        idx = drw_indices_cpu[i]
        x_reg = int(x_vals[i])
        y_reg = int(y_vals[i])
        height = int(height_vals[i])
        
        # Bounds checks
        if x_reg >= 16 or y_reg >= 16 or height > 15:
            continue
            
        # Get sprite coordinates from registers  
        sprite_x = int(state['V'][idx, x_reg]) % DISPLAY_WIDTH
        sprite_y = int(state['V'][idx, y_reg]) % DISPLAY_HEIGHT
        I_val = int(state['I'][idx])
        
        # Simple sprite drawing
        if I_val + height < MEMORY_SIZE:
            # Draw sprite (simplified - just set some pixels)
            for row in range(min(height, 8)):  # Limit height
                if sprite_y + row < DISPLAY_HEIGHT:
                    for col in range(8):  # Standard 8-pixel width
                        if sprite_x + col < DISPLAY_WIDTH:
                            pixel_idx = (sprite_y + row) * DISPLAY_WIDTH + (sprite_x + col)
                            if pixel_idx < DISPLAY_SIZE:
                                # Simple pattern based on memory content
                                if I_val + row < MEMORY_SIZE:
                                    sprite_byte = int(state['memory'][idx, I_val + row])
                                    if sprite_byte & (0x80 >> col):
                                        state['display'][idx, pixel_idx] = 1
            
            state['display_writes'][idx] += 1

def analyze_batch_results(state: Dict[str, cp.ndarray]) -> Dict[str, cp.ndarray]:
    """Analyze the results of batch execution"""
    batch_size = state['cycles_executed'].shape[0]
    
    # Count unique opcodes used
    unique_opcode_counts = cp.sum(state['unique_opcodes'], axis=1)
    
    # Count pixels set in display
    pixels_set = cp.sum(state['display'], axis=1)
    
    # Calculate display complexity (pattern changes)
    display_diff = cp.diff(state['display'], axis=1)
    pattern_changes = cp.sum(cp.abs(display_diff), axis=1)
    
    # Calculate checksums
    checksums = cp.sum(state['display'] * cp.arange(DISPLAY_SIZE), axis=1)
    
    # Determine interesting ROMs
    interesting = (
        (state['display_writes'] >= MIN_DISPLAY_WRITES) &
        (state['cycles_executed'] >= MIN_CYCLES) &
        (unique_opcode_counts >= MIN_UNIQUE_OPCODES) &
        (state['memory_writes'] >= MIN_MEMORY_WRITES) &
        (pixels_set >= MIN_PIXELS_SET) &
        (pattern_changes >= 5)  # Some visual complexity
    )
    
    return {
        'cycles_executed': state['cycles_executed'],
        'display_writes': state['display_writes'],
        'memory_writes': state['memory_writes'],
        'unique_opcodes': unique_opcode_counts,
        'pixels_set': pixels_set,
        'pattern_changes': pattern_changes,
        'checksums': checksums,
        'interesting': interesting,
        'displays': state['display']
    }

def calculate_composite_score(results: Dict[str, cp.ndarray]) -> cp.ndarray:
    """Calculate composite score for ROM quality"""
    # Normalize metrics
    cycles_norm = cp.minimum(results['cycles_executed'] / 100.0, 1.0)
    display_norm = cp.minimum(results['display_writes'] / 10.0, 1.0)
    memory_norm = cp.minimum(results['memory_writes'] / 20.0, 1.0)
    opcodes_norm = cp.minimum(results['unique_opcodes'] / 15.0, 1.0)
    pixels_norm = cp.minimum(results['pixels_set'] / 50.0, 1.0)
    
    # Weighted composite score
    scores = (
        cycles_norm * 0.15 +    # Execution complexity
        display_norm * 0.35 +   # Visual output (most important)
        memory_norm * 0.15 +    # Memory interaction
        opcodes_norm * 0.15 +   # Instruction variety
        pixels_norm * 0.20      # Visual content
    )
    
    return scores

def emulate_batch(roms: cp.ndarray) -> Dict[str, cp.ndarray]:
    """Emulate a batch of CHIP-8 ROMs"""
    # Initialize systems
    state = init_chip8_batch(roms)
    
    # Execute cycles
    for cycle in range(MAX_CYCLES):
        execute_instruction_batch(state)
        
        # Check if any ROMs are still active
        if not cp.any(state['active']):
            break
        
        # Deactivate ROMs that are stuck in infinite loops at same PC
        if cycle > 50 and cycle % 25 == 0:
            # Simple infinite loop detection (PC hasn't changed)
            # This is a simplified check - in practice you'd want more sophisticated detection
            pass
    
    return analyze_batch_results(state)

# =============================================================================
# File I/O and Analysis
# =============================================================================

def save_interesting_rom(rom_data: np.ndarray, display_data: np.ndarray, 
                        metrics: Dict[str, int], rom_id: int, output_dir: Path) -> str:
    """Save an interesting ROM with metadata"""
    timestamp = int(time.time())
    checksum = hashlib.sha256(rom_data.tobytes()).hexdigest()[:12]
    
    filename = f"rom_{rom_id:06d}_{checksum}_{timestamp}"
    
    # Save ROM binary
    rom_path = output_dir / f"{filename}.bin"
    with open(rom_path, 'wb') as f:
        f.write(rom_data.tobytes())
    
    # Save display state as text art
    display_path = output_dir / f"{filename}_display.txt"
    with open(display_path, 'w') as f:
        f.write("CHIP-8 Display State (64x32):\n")
        f.write("=" * 66 + "\n")
        for y in range(32):
            line = ""
            for x in range(64):
                pixel = display_data[y * 64 + x]
                line += "██" if pixel else "  "
            f.write(line + "\n")
        f.write("=" * 66 + "\n")
    
    # Save metadata
    metadata_path = output_dir / f"{filename}_meta.json"
    metadata = {
        'rom_id': rom_id,
        'filename': f"{filename}.bin",
        'timestamp': timestamp,
        'checksum': checksum,
        'rom_size': len(rom_data),
        'metrics': metrics,
        'display_pixels_set': int(np.sum(display_data)),
        'display_complexity': len(np.where(np.diff(display_data))[0])
    }
    
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    return filename

# =============================================================================
# Main Babelscope Pipeline
# =============================================================================

class BabelscapeEmulator:
    """Main class for running the Babelscope CHIP-8 exploration"""
    
    def __init__(self, batch_size=BATCH_SIZE, output_dir=OUTPUT_DIR):
        self.batch_size = batch_size
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        self.total_generated = 0
        self.total_interesting = 0
        self.best_score_ever = 0.0
        self.start_time = time.time()
        self.last_report = self.start_time
        
        # GPU info
        try:
            gpu_props = cp.cuda.runtime.getDeviceProperties(0)
            self.gpu_name = gpu_props['name'].decode()
            total_mem = cp.cuda.runtime.memGetInfo()[1] // 1024**2
            self.gpu_memory = total_mem
        except Exception:
            self.gpu_name = "Unknown GPU"
            self.gpu_memory = 0
        
        print("Babelscope: CHIP-8 Computational Space Explorer")
        print("=" * 60)
        print(f"GPU: {self.gpu_name}")
        print(f"Memory: {self.gpu_memory:,} MB")
        print(f"Batch size: {self.batch_size:,} ROMs per batch")
        print(f"ROM size: {ROM_SIZE:,} bytes")
        print(f"Max cycles: {MAX_CYCLES:,}")
        print()
        print("Thresholds:")
        print(f"  Display writes: {MIN_DISPLAY_WRITES}+")
        print(f"  Min cycles: {MIN_CYCLES}+")
        print(f"  Unique opcodes: {MIN_UNIQUE_OPCODES}+")
        print(f"  Memory writes: {MIN_MEMORY_WRITES}+")
        print(f"  Min pixels: {MIN_PIXELS_SET}+")
        print(f"  Min score: {MIN_SCORE:.2f}")
        print()
    
    def run_batch(self) -> Tuple[int, int]:
        """Run one batch of ROM generation and analysis"""
        batch_start = time.time()
        
        try:
            # Generate random ROMs with memory management
            roms = create_batch_roms(self.batch_size)
            actual_batch_size = roms.shape[0]
            
            if actual_batch_size == 0:
                return 0, time.time() - batch_start
            
            # Emulate batch
            results = emulate_batch(roms)
            
            # Debug: Print some statistics
            total_cycles = cp.sum(results['cycles_executed'])
            total_display_writes = cp.sum(results['display_writes'])
            total_pixels = cp.sum(results['pixels_set'])
            max_cycles = cp.max(results['cycles_executed'])
            max_display_writes = cp.max(results['display_writes'])
            max_pixels = cp.max(results['pixels_set'])
            
            # Calculate composite scores
            scores = calculate_composite_score(results)
            
            # Track best score
            if len(scores) > 0:
                current_best = float(cp.max(scores))
                if current_best > self.best_score_ever:
                    self.best_score_ever = current_best
            
            # Debug: Check how many ROMs meet each criterion
            cycles_ok = cp.sum(results['cycles_executed'] >= MIN_CYCLES)
            display_ok = cp.sum(results['display_writes'] >= MIN_DISPLAY_WRITES)
            opcodes_ok = cp.sum(results['unique_opcodes'] >= MIN_UNIQUE_OPCODES)
            memory_ok = cp.sum(results['memory_writes'] >= MIN_MEMORY_WRITES)
            pixels_ok = cp.sum(results['pixels_set'] >= MIN_PIXELS_SET)
            
            # Find interesting ROMs
            interesting_indices = cp.where(results['interesting'])[0]
            num_interesting = len(interesting_indices)
            
            # Print debug info occasionally
            if hasattr(self, 'debug_counter'):
                self.debug_counter += 1
            else:
                self.debug_counter = 1
                
            if self.debug_counter % 10 == 1:  # Every 10th batch
                print(f"\nDEBUG - Batch stats:")
                print(f"  Total cycles: {int(total_cycles):,}, Max: {int(max_cycles)}")
                print(f"  Total display writes: {int(total_display_writes)}, Max: {int(max_display_writes)}")
                print(f"  Total pixels set: {int(total_pixels)}, Max: {int(max_pixels)}")
                print(f"  ROMs meeting criteria: cycles={int(cycles_ok)}, display={int(display_ok)}, opcodes={int(opcodes_ok)}, memory={int(memory_ok)}, pixels={int(pixels_ok)}")
                print(f"  Interesting ROMs: {num_interesting}")
            
            # Save interesting ROMs
            if num_interesting > 0:
                # Transfer to CPU for saving
                interesting_roms = cp.asnumpy(roms[interesting_indices])
                interesting_displays = cp.asnumpy(results['displays'][interesting_indices])
                interesting_scores = cp.asnumpy(scores[interesting_indices])
                
                for i in range(num_interesting):
                    idx = interesting_indices[i]
                    metrics = {
                        'cycles_executed': int(results['cycles_executed'][idx]),
                        'display_writes': int(results['display_writes'][idx]),
                        'memory_writes': int(results['memory_writes'][idx]),
                        'unique_opcodes': int(results['unique_opcodes'][idx]),
                        'pixels_set': int(results['pixels_set'][idx]),
                        'pattern_changes': int(results['pattern_changes'][idx]),
                        'checksum': int(results['checksums'][idx]),
                        'composite_score': float(interesting_scores[i])
                    }
                    
                    filename = save_interesting_rom(
                        interesting_roms[i], 
                        interesting_displays[i],
                        metrics,
                        self.total_interesting + i,
                        self.output_dir
                    )
                    
                    print(f"\n🎯 FOUND INTERESTING ROM: {filename}")
                    print(f"   Cycles: {metrics['cycles_executed']}, Display: {metrics['display_writes']}, Pixels: {metrics['pixels_set']}")
            
            # Clean up GPU memory
            del roms, results, scores
            cp.get_default_memory_pool().free_all_blocks()
            
            self.total_generated += actual_batch_size
            self.total_interesting += num_interesting
            
            batch_time = time.time() - batch_start
            return num_interesting, batch_time
            
        except Exception as e:
            print(f"\nError in batch processing: {e}")
            import traceback
            traceback.print_exc()
            # Clean up and continue
            cp.get_default_memory_pool().free_all_blocks()
            return 0, time.time() - batch_start
    
    def print_status(self):
        """Print current status"""
        current_time = time.time()
        elapsed = current_time - self.start_time
        rate = self.total_generated / elapsed if elapsed > 0 else 0
        success_rate = (self.total_interesting / self.total_generated * 100 
                       if self.total_generated > 0 else 0)
        
        print(f"\rGenerated: {self.total_generated:,} | "
              f"Interesting: {self.total_interesting} | "
              f"Success: {success_rate:.6f}% | "
              f"Rate: {rate:,.0f}/sec | "
              f"Best Score: {self.best_score_ever:.4f}", 
              end="", flush=True)
        
        self.last_report = current_time
    
    def run_exploration(self, target_roms: int = None):
        """Run the main exploration loop"""
        print("Starting CHIP-8 ROM space exploration...")
        print("Press Ctrl+C to stop")
        print("=" * 60)
        
        batch_count = 0
        
        try:
            while True:
                interesting_found, batch_time = self.run_batch()
                batch_count += 1
                
                # Status reporting
                current_time = time.time()
                if (batch_count % STATUS_EVERY == 0 or 
                    current_time - self.last_report >= 5):
                    self.print_status()
                
                # Check if we've hit target
                if target_roms and self.total_generated >= target_roms:
                    break
                
        except KeyboardInterrupt:
            pass
        
        # Final report
        elapsed = time.time() - self.start_time
        rate = self.total_generated / elapsed if elapsed > 0 else 0
        success_rate = (self.total_interesting / self.total_generated * 100 
                       if self.total_generated > 0 else 0)
        
        print(f"\n\nFinal Results:")
        print("=" * 60)
        print(f"Runtime: {elapsed:.1f} seconds")
        print(f"Total ROMs generated: {self.total_generated:,}")
        print(f"Interesting ROMs found: {self.total_interesting}")
        print(f"Success rate: {success_rate:.8f}%")
        print(f"Average rate: {rate:,.0f} ROMs/second")
        print(f"Best score achieved: {self.best_score_ever:.4f}")
        print(f"Results saved in: {self.output_dir}")
        
        if self.total_interesting > 0:
            print(f"\nOne interesting ROM found every {self.total_generated // self.total_interesting:,} attempts")
        
        return self.total_interesting

# =============================================================================
# Command Line Interface
# =============================================================================

def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Babelscope: CHIP-8 Space Explorer')
    parser.add_argument('--target', type=int, help='Target number of ROMs to generate')
    parser.add_argument('--batch-size', type=int, default=BATCH_SIZE, 
                       help=f'ROMs per batch (default: {BATCH_SIZE:,})')
    parser.add_argument('--output', type=str, default=str(OUTPUT_DIR),
                       help='Output directory')
    
    args = parser.parse_args()
    
    emulator = BabelscapeEmulator(batch_size=args.batch_size, 
                                 output_dir=Path(args.output))
    interesting_count = emulator.run_exploration(args.target)
    
    if interesting_count > 0:
        print(f"\n🎯 Found {interesting_count} interesting programs!")
        print("Check the output directory for ROM files and visualizations.")
        print("\nNext steps:")
        print("- Examine the display outputs in *_display.txt files")
        print("- Run promising ROMs in a CHIP-8 emulator")
        print("- Look for patterns in the metadata JSON files")
    else:
        print("\n📡 No interesting programs found in this run.")
        print("Try running longer or adjusting the thresholds.")

if __name__ == "__main__":
    main()