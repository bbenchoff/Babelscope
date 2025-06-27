"""
Enhanced Babelscope Implementation: Partial Register Sorting Detection
Now detects partial sorting sequences (3+ consecutive elements) in registers V0-V7
This dramatically increases discovery probability by detecting incremental progress!

OPTIMIZED VERSION: Reduced overhead, faster execution, cleaner output
"""

import cupy as cp
import numpy as np
from typing import List, Dict, Optional, Tuple
import time
import hashlib
from pathlib import Path
import json

# CHIP-8 Constants
MEMORY_SIZE = 4096
DISPLAY_WIDTH = 64
DISPLAY_HEIGHT = 32
REGISTER_COUNT = 16
STACK_SIZE = 16
KEYPAD_SIZE = 16
PROGRAM_START = 0x200
FONT_START = 0x50

# Partial sorting detection constants
SORT_ARRAY_SIZE = 8
SORT_REGISTERS = list(range(8))  # V0 through V7
MIN_PARTIAL_LENGTH = 3  # Minimum consecutive sorted elements to detect
MIN_SAVE_LENGTH = 6     # Minimum length to save to disk (only save 6+ element sequences)

print(f"Partial sorting setup: Detecting {MIN_PARTIAL_LENGTH}+ consecutive sorted elements")
print(f"Saving only sequences of length {MIN_SAVE_LENGTH}+ (6, 7, or 8 elements)")

# Font data (must be loaded into all instances)
CHIP8_FONT = cp.array([
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

# OPTIMIZED Enhanced CHIP-8 emulation kernel with partial sorting detection
# Performance improvements: reduced branching, faster sorting check, optimized memory access
OPTIMIZED_CHIP8_KERNEL = r'''
extern "C" __global__ __launch_bounds__(256, 4)
void chip8_partial_sorting_kernel(
    // Core CHIP-8 state
    unsigned char* __restrict__ memory,              // [instances][4096]
    unsigned char* __restrict__ display,             // [instances][32*64]
    unsigned char* __restrict__ registers,           // [instances][16]
    unsigned short* __restrict__ index_registers,    // [instances]
    unsigned short* __restrict__ program_counters,   // [instances]
    unsigned char* __restrict__ stack_pointers,      // [instances]
    unsigned short* __restrict__ stacks,             // [instances][16]
    unsigned char* __restrict__ delay_timers,        // [instances]
    unsigned char* __restrict__ sound_timers,        // [instances]
    unsigned char* __restrict__ keypad,              // [instances][16]
    
    // State flags
    unsigned char* __restrict__ crashed,             // [instances]
    unsigned char* __restrict__ halted,              // [instances]
    unsigned char* __restrict__ waiting_for_key,     // [instances]
    unsigned char* __restrict__ key_registers,       // [instances]
    
    // Partial sorting detection arrays
    unsigned int* __restrict__ sort_cycles,          // [instances] - cycle when sorted
    unsigned char* __restrict__ sort_achieved,       // [instances] - 1 if any sorting found
    unsigned char* __restrict__ sort_lengths,        // [instances] - length of longest sorted sequence
    unsigned char* __restrict__ sort_start_positions,// [instances] - start position of best sequence
    unsigned char* __restrict__ sort_directions,     // [instances] - 0=ascending, 1=descending
    
    // Register access tracking
    unsigned int* __restrict__ total_register_ops,   // [instances]
    unsigned int* __restrict__ register_reads,       // [instances]
    unsigned int* __restrict__ register_writes,      // [instances]
    
    // Execution parameters
    int num_instances,
    int cycles_to_run,
    int sort_check_interval,
    int min_partial_length,
    
    // RNG state
    unsigned int* __restrict__ rng_state
) {
    const int instance = blockIdx.x * blockDim.x + threadIdx.x;
    if (instance >= num_instances) return;
    
    // Calculate base offsets
    const int mem_base = instance * 4096;
    const int display_base = instance * 32 * 64;
    const int reg_base = instance * 16;
    const int stack_base = instance * 16;
    const int keypad_base = instance * 16;
    
    // Load state into registers for performance
    unsigned short pc = program_counters[instance];
    unsigned short index_reg = index_registers[instance];
    unsigned char sp = stack_pointers[instance];
    unsigned char dt = delay_timers[instance];
    unsigned char st = sound_timers[instance];
    
    // Local counters
    unsigned int local_register_ops = 0;
    unsigned int local_register_reads = 0;
    unsigned int local_register_writes = 0;
    
    // Early exit if crashed or already found sorting
    if (crashed[instance] || sort_achieved[instance]) {
        return;
    }
    
    // Cache register values for faster sorting checks
    unsigned char reg_cache[8];
    bool cache_valid = false;
    
    // Main execution loop
    for (int cycle = 0; cycle < cycles_to_run; cycle++) {
        // Skip if waiting for key (we don't simulate key input)
        if (waiting_for_key[instance]) {
            continue;
        }
        
        // Bounds check
        if (pc >= 4094) {
            crashed[instance] = 1;
            break;
        }
        
        // Fetch instruction
        const unsigned char high_byte = memory[mem_base + pc];
        const unsigned char low_byte = memory[mem_base + pc + 1];
        const unsigned short instruction = (high_byte << 8) | low_byte;
        
        pc += 2;
        
        // Decode instruction
        const unsigned char opcode = (instruction & 0xF000) >> 12;
        const unsigned char x = (instruction & 0x0F00) >> 8;
        const unsigned char y = (instruction & 0x00F0) >> 4;
        const unsigned char n = instruction & 0x000F;
        const unsigned char kk = instruction & 0x00FF;
        const unsigned short nnn = instruction & 0x0FFF;
        
        // Track if we modify registers V0-V7 for cache invalidation
        bool invalidate_cache = false;
        
        // Execute instruction - COMPLETE CHIP-8 IMPLEMENTATION
        switch (opcode) {
            case 0x0:
                if (instruction == 0x00E0) {
                    // CLS - Clear display
                    for (int i = 0; i < 32 * 64; i++) {
                        display[display_base + i] = 0;
                    }
                } else if (instruction == 0x00EE) {
                    // RET - Return from subroutine
                    if (sp > 0) {
                        sp--;
                        pc = stacks[stack_base + sp];
                    } else {
                        crashed[instance] = 1;
                    }
                }
                break;
                
            case 0x1:
                // JP addr - Jump to address
                pc = nnn;
                break;
                
            case 0x2:
                // CALL addr - Call subroutine
                if (sp < 16) {
                    stacks[stack_base + sp] = pc;
                    sp++;
                    pc = nnn;
                } else {
                    crashed[instance] = 1;
                }
                break;
                
            case 0x3:
                // SE Vx, byte - Skip if Vx == byte
                local_register_reads++;
                if (registers[reg_base + x] == kk) {
                    pc += 2;
                }
                break;
                
            case 0x4:
                // SNE Vx, byte - Skip if Vx != byte
                local_register_reads++;
                if (registers[reg_base + x] != kk) {
                    pc += 2;
                }
                break;
                
            case 0x5:
                // SE Vx, Vy - Skip if Vx == Vy
                if (n == 0) {
                    local_register_reads += 2;
                    if (registers[reg_base + x] == registers[reg_base + y]) {
                        pc += 2;
                    }
                }
                break;
                
            case 0x6:
                // LD Vx, byte - Load byte into Vx
                registers[reg_base + x] = kk;
                local_register_writes++;
                if (x < 8) invalidate_cache = true;
                break;
                
            case 0x7:
                // ADD Vx, byte - Add byte to Vx
                local_register_reads++;
                local_register_writes++;
                registers[reg_base + x] = (registers[reg_base + x] + kk) & 0xFF;
                if (x < 8) invalidate_cache = true;
                break;
                
            case 0x8:
                // Register operations
                {
                    const unsigned char vx = registers[reg_base + x];
                    const unsigned char vy = registers[reg_base + y];
                    local_register_reads += 2;
                    
                    switch (n) {
                        case 0x0: // LD Vx, Vy
                            registers[reg_base + x] = vy;
                            local_register_writes++;
                            if (x < 8) invalidate_cache = true;
                            break;
                        case 0x1: // OR Vx, Vy
                            registers[reg_base + x] = vx | vy;
                            registers[reg_base + 0xF] = 0;
                            local_register_writes += 2;
                            if (x < 8) invalidate_cache = true;
                            break;
                        case 0x2: // AND Vx, Vy
                            registers[reg_base + x] = vx & vy;
                            registers[reg_base + 0xF] = 0;
                            local_register_writes += 2;
                            if (x < 8) invalidate_cache = true;
                            break;
                        case 0x3: // XOR Vx, Vy
                            registers[reg_base + x] = vx ^ vy;
                            registers[reg_base + 0xF] = 0;
                            local_register_writes += 2;
                            if (x < 8) invalidate_cache = true;
                            break;
                        case 0x4: // ADD Vx, Vy
                            {
                                const int result = vx + vy;
                                registers[reg_base + x] = result & 0xFF;
                                registers[reg_base + 0xF] = (result > 255) ? 1 : 0;
                                local_register_writes += 2;
                                if (x < 8) invalidate_cache = true;
                            }
                            break;
                        case 0x5: // SUB Vx, Vy
                            registers[reg_base + x] = (vx - vy) & 0xFF;
                            registers[reg_base + 0xF] = (vx >= vy) ? 1 : 0;
                            local_register_writes += 2;
                            if (x < 8) invalidate_cache = true;
                            break;
                        case 0x6: // SHR Vx
                            registers[reg_base + x] = vx >> 1;
                            registers[reg_base + 0xF] = vx & 0x1;
                            local_register_writes += 2;
                            if (x < 8) invalidate_cache = true;
                            break;
                        case 0x7: // SUBN Vx, Vy
                            registers[reg_base + x] = (vy - vx) & 0xFF;
                            registers[reg_base + 0xF] = (vy >= vx) ? 1 : 0;
                            local_register_writes += 2;
                            if (x < 8) invalidate_cache = true;
                            break;
                        case 0xE: // SHL Vx
                            registers[reg_base + x] = (vx << 1) & 0xFF;
                            registers[reg_base + 0xF] = (vx & 0x80) ? 1 : 0;
                            local_register_writes += 2;
                            if (x < 8) invalidate_cache = true;
                            break;
                        default:
                            crashed[instance] = 1;
                            break;
                    }
                    local_register_ops++;
                }
                break;
                
            case 0x9:
                // SNE Vx, Vy - Skip if Vx != Vy
                if (n == 0) {
                    local_register_reads += 2;
                    if (registers[reg_base + x] != registers[reg_base + y]) {
                        pc += 2;
                    }
                }
                break;
                
            case 0xA:
                // LD I, addr - Load address into I
                index_reg = nnn;
                break;
                
            case 0xB:
                // JP V0, addr - Jump to V0 + addr
                local_register_reads++;
                pc = nnn + registers[reg_base + 0];
                break;
                
            case 0xC:
                // RND Vx, byte - Random number AND byte
                {
                    rng_state[instance] = rng_state[instance] * 1664525 + 1013904223;
                    const unsigned char rand_byte = (rng_state[instance] >> 16) & 0xFF;
                    registers[reg_base + x] = rand_byte & kk;
                    local_register_writes++;
                    if (x < 8) invalidate_cache = true;
                }
                break;
                
            case 0xD:
                // DRW Vx, Vy, nibble - Draw sprite
                {
                    const unsigned char start_x = registers[reg_base + x] % 64;
                    const unsigned char start_y = registers[reg_base + y] % 32;
                    local_register_reads += 2;
                    registers[reg_base + 0xF] = 0;
                    local_register_writes++;
                    
                    for (int row = 0; row < n && start_y + row < 32; row++) {
                        if (index_reg + row >= 4096) break;
                        
                        const unsigned char sprite_byte = memory[mem_base + index_reg + row];
                        
                        for (int col = 0; col < 8 && start_x + col < 64; col++) {
                            if (sprite_byte & (0x80 >> col)) {
                                const int pixel_index = display_base + (start_y + row) * 64 + (start_x + col);
                                
                                if (display[pixel_index]) {
                                    registers[reg_base + 0xF] = 1;
                                    local_register_writes++;
                                }
                                
                                display[pixel_index] ^= 1;
                            }
                        }
                    }
                }
                break;
                
            case 0xE:
                // Key operations
                {
                    const unsigned char key = registers[reg_base + x] & 0xF;
                    local_register_reads++;
                    if (kk == 0x9E) {
                        // SKP Vx - Skip if key pressed
                        if (keypad[keypad_base + key]) {
                            pc += 2;
                        }
                    } else if (kk == 0xA1) {
                        // SKNP Vx - Skip if key not pressed
                        if (!keypad[keypad_base + key]) {
                            pc += 2;
                        }
                    } else {
                        crashed[instance] = 1;
                    }
                }
                break;
                
            case 0xF:
                // Misc operations
                switch (kk) {
                    case 0x07: // LD Vx, DT
                        registers[reg_base + x] = dt;
                        local_register_writes++;
                        if (x < 8) invalidate_cache = true;
                        break;
                    case 0x0A: // LD Vx, K - Wait for key
                        waiting_for_key[instance] = 1;
                        key_registers[instance] = x;
                        break;
                    case 0x15: // LD DT, Vx
                        dt = registers[reg_base + x];
                        local_register_reads++;
                        break;
                    case 0x18: // LD ST, Vx
                        st = registers[reg_base + x];
                        local_register_reads++;
                        break;
                    case 0x1E: // ADD I, Vx
                        index_reg = (index_reg + registers[reg_base + x]) & 0xFFFF;
                        local_register_reads++;
                        break;
                    case 0x29: // LD F, Vx - Set I to font location
                        {
                            const unsigned char digit = registers[reg_base + x] & 0xF;
                            local_register_reads++;
                            index_reg = 0x50 + digit * 5;
                        }
                        break;
                    case 0x33: // LD B, Vx - Store BCD
                        {
                            const unsigned char value = registers[reg_base + x];
                            local_register_reads++;
                            if (index_reg + 2 < 4096) {
                                memory[mem_base + index_reg] = value / 100;
                                memory[mem_base + index_reg + 1] = (value / 10) % 10;
                                memory[mem_base + index_reg + 2] = value % 10;
                            }
                        }
                        break;
                    case 0x55: // LD [I], Vx - Store registers
                        for (int i = 0; i <= x; i++) {
                            if (index_reg + i < 4096) {
                                memory[mem_base + index_reg + i] = registers[reg_base + i];
                                local_register_reads++;
                            }
                        }
                        index_reg = (index_reg + x + 1) & 0xFFFF;
                        break;
                    case 0x65: // LD Vx, [I] - Load registers
                        for (int i = 0; i <= x; i++) {
                            if (index_reg + i < 4096) {
                                registers[reg_base + i] = memory[mem_base + index_reg + i];
                                local_register_writes++;
                                if (i < 8) invalidate_cache = true;
                            }
                        }
                        index_reg = (index_reg + x + 1) & 0xFFFF;
                        break;
                    default:
                        crashed[instance] = 1;
                        break;
                }
                break;
                
            default:
                crashed[instance] = 1;
                break;
        }
        
        // Invalidate cache if V0-V7 registers were modified
        if (invalidate_cache) {
            cache_valid = false;
        }
        
        // OPTIMIZED: Check for partial sorting every N cycles
        if ((cycle % sort_check_interval) == 0 && !sort_achieved[instance]) {
            // Update cache if needed
            if (!cache_valid) {
                for (int i = 0; i < 8; i++) {
                    reg_cache[i] = registers[reg_base + i];
                }
                cache_valid = true;
            }
            
            // Find longest consecutive sorted sequence in either direction
            unsigned char best_length = 0;
            unsigned char best_start = 0;
            unsigned char best_direction = 0; // 0=ascending, 1=descending
            
            // OPTIMIZED: Check ascending sequences with early termination
            for (int start = 0; start <= 8 - min_partial_length; start++) {
                unsigned char length = 1;
                for (int i = start; i < 7; i++) {
                    if (reg_cache[i] + 1 == reg_cache[i + 1]) {
                        length++;
                    } else {
                        break;
                    }
                }
                
                if (length >= min_partial_length && length > best_length) {
                    best_length = length;
                    best_start = start;
                    best_direction = 0;
                    
                    // Early exit if we found maximum possible length
                    if (length == 8) break;
                }
            }
            
            // OPTIMIZED: Check descending sequences with early termination
            for (int start = 0; start <= 8 - min_partial_length; start++) {
                unsigned char length = 1;
                for (int i = start; i < 7; i++) {
                    if (reg_cache[i] == reg_cache[i + 1] + 1) {
                        length++;
                    } else {
                        break;
                    }
                }
                
                if (length >= min_partial_length && length > best_length) {
                    best_length = length;
                    best_start = start;
                    best_direction = 1;
                    
                    // Early exit if we found maximum possible length
                    if (length == 8) break;
                }
            }
            
            // Record discovery if we found a valid sequence
            if (best_length >= min_partial_length) {
                sort_achieved[instance] = 1;
                sort_cycles[instance] = cycle;
                sort_lengths[instance] = best_length;
                sort_start_positions[instance] = best_start;
                sort_directions[instance] = best_direction;
                break; // Found one! Early termination
            }
        }
        
        // Update timers periodically (less frequently for performance)
        if ((cycle & 31) == 0) {
            if (dt > 0) dt--;
            if (st > 0) st--;
        }
    }
    
    // Write back state
    program_counters[instance] = pc;
    index_registers[instance] = index_reg;
    stack_pointers[instance] = sp;
    delay_timers[instance] = dt;
    sound_timers[instance] = st;
    
    // Write back access counts
    total_register_ops[instance] += local_register_ops;
    register_reads[instance] += local_register_reads;
    register_writes[instance] += local_register_writes;
}
'''

class PartialSortingBabelscopeDetector:
    """
    Partial Sorting Babelscope: Detects 3+ consecutive sorted elements in registers V0-V7
    OPTIMIZED VERSION: Faster execution, reduced overhead, cleaner output
    """
    
    def __init__(self, num_instances: int):
        print(f"Initializing Partial Sorting Babelscope Detector (OPTIMIZED)")
        print(f"Instances: {num_instances:,}")
        print(f"Monitoring registers V0-V7 for sorting")
        
        self.num_instances = num_instances
        
        # Calculate grid configuration
        self.block_size = 256
        self.grid_size = (num_instances + self.block_size - 1) // self.block_size
        
        # Compile the optimized CHIP-8 kernel
        try:
            device = cp.cuda.Device()
            device_props = cp.cuda.runtime.getDeviceProperties(device.id)
            compute_capability = device.compute_capability
            
            # Fix compute capability formatting for RTX 5080 (12.0 -> sm_89)
            # RTX 5080 should use sm_89 architecture
            if compute_capability >= (12, 0):
                arch_flag = '--gpu-architecture=sm_89'
            elif compute_capability >= (9, 0):
                arch_flag = '--gpu-architecture=sm_90'
            elif compute_capability >= (8, 0):
                arch_flag = '--gpu-architecture=sm_80'
            elif compute_capability >= (7, 5):
                arch_flag = '--gpu-architecture=sm_75'
            elif compute_capability >= (7, 0):
                arch_flag = '--gpu-architecture=sm_70'
            else:
                arch_flag = '--gpu-architecture=sm_60'
            
            print(f"Detected compute capability {compute_capability}, using {arch_flag}")
            
            self.kernel = cp.RawKernel(
                OPTIMIZED_CHIP8_KERNEL, 
                'chip8_partial_sorting_kernel',
                options=(arch_flag,)
            )
            print(f"Kernel compiled successfully with {arch_flag}")
        except Exception as e:
            print(f"GPU-specific compilation failed: {e}")
            # Try with just fast math
            try:
                self.kernel = cp.RawKernel(
                    OPTIMIZED_CHIP8_KERNEL, 
                    'chip8_partial_sorting_kernel',
                    options=('--use_fast_math',)
                )
                print("Kernel compiled with fast math")
            except Exception as e2:
                print(f"Fast math compilation failed: {e2}")
                # Last resort - no options at all
                try:
                    self.kernel = cp.RawKernel(OPTIMIZED_CHIP8_KERNEL, 'chip8_partial_sorting_kernel')
                    print("Kernel compiled with no optimizations")
                except Exception as e3:
                    print(f"CRITICAL: Kernel compilation completely failed: {e3}")
                    raise
        
        # Initialize state
        self._initialize_state()
        
        print("Partial Sorting Babelscope ready")
    
    def _initialize_state(self):
        """Initialize all GPU arrays with optimized memory layout"""
        
        # Core CHIP-8 state - use pinned memory for faster transfers
        self.memory = cp.zeros((self.num_instances, MEMORY_SIZE), dtype=cp.uint8)
        self.display = cp.zeros((self.num_instances, DISPLAY_HEIGHT * DISPLAY_WIDTH), dtype=cp.uint8)
        self.registers = cp.zeros((self.num_instances, REGISTER_COUNT), dtype=cp.uint8)
        self.index_register = cp.zeros(self.num_instances, dtype=cp.uint16)
        self.program_counter = cp.full(self.num_instances, PROGRAM_START, dtype=cp.uint16)
        self.stack_pointer = cp.zeros(self.num_instances, dtype=cp.uint8)
        self.stack = cp.zeros((self.num_instances, STACK_SIZE), dtype=cp.uint16)
        self.delay_timer = cp.zeros(self.num_instances, dtype=cp.uint8)
        self.sound_timer = cp.zeros(self.num_instances, dtype=cp.uint8)
        self.keypad = cp.zeros((self.num_instances, KEYPAD_SIZE), dtype=cp.uint8)
        
        # State flags
        self.crashed = cp.zeros(self.num_instances, dtype=cp.uint8)
        self.halted = cp.zeros(self.num_instances, dtype=cp.uint8)
        self.waiting_for_key = cp.zeros(self.num_instances, dtype=cp.uint8)
        self.key_register = cp.zeros(self.num_instances, dtype=cp.uint8)
        
        # Partial sorting detection
        self.sort_cycles = cp.zeros(self.num_instances, dtype=cp.uint32)
        self.sort_achieved = cp.zeros(self.num_instances, dtype=cp.uint8)
        self.sort_lengths = cp.zeros(self.num_instances, dtype=cp.uint8)
        self.sort_start_positions = cp.zeros(self.num_instances, dtype=cp.uint8)
        self.sort_directions = cp.zeros(self.num_instances, dtype=cp.uint8)
        
        # Register access tracking
        self.total_register_ops = cp.zeros(self.num_instances, dtype=cp.uint32)
        self.register_reads = cp.zeros(self.num_instances, dtype=cp.uint32)
        self.register_writes = cp.zeros(self.num_instances, dtype=cp.uint32)
        
        # RNG state
        self.rng_state = cp.random.randint(1, 2**32, size=self.num_instances, dtype=cp.uint32)
        
        # Load font into all instances - optimized batch operation
        font_data = cp.tile(CHIP8_FONT, (self.num_instances, 1))
        self.memory[:, FONT_START:FONT_START + len(CHIP8_FONT)] = font_data
    
    def load_random_roms_and_setup_register_test(self, rom_data):
        """Load random ROMs and setup register-based test for partial sorting"""
        
        if isinstance(rom_data, cp.ndarray):
            print(f"Loading {rom_data.shape[0]:,} random ROMs from GPU array...")
            
            num_roms_to_load = min(rom_data.shape[0], self.num_instances)
            rom_size = min(rom_data.shape[1], MEMORY_SIZE - PROGRAM_START)
            rom_end = PROGRAM_START + rom_size
            
            # Optimized memory copy
            self.memory[:num_roms_to_load, PROGRAM_START:rom_end] = rom_data[:num_roms_to_load, :rom_size]
            
            if num_roms_to_load < self.num_instances:
                # Batch tile remaining instances
                remaining = self.num_instances - num_roms_to_load
                repeats = (remaining + num_roms_to_load - 1) // num_roms_to_load
                
                for rep in range(repeats):
                    start_idx = num_roms_to_load + rep * num_roms_to_load
                    end_idx = min(start_idx + num_roms_to_load, self.num_instances)
                    copy_size = end_idx - start_idx
                    
                    self.memory[start_idx:end_idx, PROGRAM_START:rom_end] = rom_data[:copy_size, :rom_size]
        else:
            print(f"Loading {len(rom_data):,} random ROMs from CPU list...")
            
            for i in range(self.num_instances):
                rom_array = rom_data[i % len(rom_data)]
                
                if len(rom_array) > MEMORY_SIZE - PROGRAM_START:
                    rom_array = rom_array[:MEMORY_SIZE - PROGRAM_START]
                
                rom_end = PROGRAM_START + len(rom_array)
                self.memory[i, PROGRAM_START:rom_end] = cp.array(rom_array)
        
        # Setup the test pattern in REGISTERS V0-V7
        test_pattern = np.array([8, 3, 6, 1, 7, 2, 5, 4], dtype=np.uint8)
        
        print(f"Setting up partial sorting test:")
        print(f"   Test pattern: {test_pattern}")
        print(f"   Target registers: V0-V7")
        print(f"   Detection: {MIN_PARTIAL_LENGTH}+ consecutive sorted elements")
        print(f"   Examples: [1,2,3] or [8,7,6] or [1,2,3,4] etc.")
        
        # Place test pattern in registers V0-V7 for all instances - optimized batch operation
        test_pattern_gpu = cp.array(test_pattern)
        self.registers[:, :8] = test_pattern_gpu[None, :]
        
        print(f"   Loaded test pattern into registers V0-V7")
        print(f"   Random code space: 3584 bytes (0x200-0xFFF)")
        print(f"   Enhancement: Detects incremental sorting progress!")
    
    def run_partial_sorting_search(self, cycles: int = 100000, check_interval: int = 100) -> int:
        """Run the optimized partial sorting Babelscope search"""
        
        start_time = time.time()
        
        # Launch the optimized partial sorting CHIP-8 kernel
        self.kernel(
            (self.grid_size,), (self.block_size,),
            (
                # Core CHIP-8 state
                self.memory,
                self.display,
                self.registers,
                self.index_register,
                self.program_counter,
                self.stack_pointer,
                self.stack,
                self.delay_timer,
                self.sound_timer,
                self.keypad,
                
                # State flags
                self.crashed,
                self.halted,
                self.waiting_for_key,
                self.key_register,
                
                # Partial sorting detection
                self.sort_cycles,
                self.sort_achieved,
                self.sort_lengths,
                self.sort_start_positions,
                self.sort_directions,
                
                # Register access tracking
                self.total_register_ops,
                self.register_reads,
                self.register_writes,
                
                # Parameters
                self.num_instances,
                cycles,
                check_interval,
                MIN_PARTIAL_LENGTH,
                
                # RNG
                self.rng_state
            )
        )
        
        # Synchronize and get results
        cp.cuda.Stream.null.synchronize()
        execution_time = time.time() - start_time
        
        # Count results
        sorts_found = int(cp.sum(self.sort_achieved))
        
        return sorts_found
    
    def get_partial_sorting_discoveries(self) -> List[Dict]:
        """Get all discovered partial sorting algorithms with metadata"""
        discoveries = []
        
        # Find instances that achieved partial sorting
        sorted_indices = cp.where(self.sort_achieved)[0]
        
        for idx in sorted_indices:
            idx = int(idx)
            
            # Get the sorted registers V0-V7
            final_registers = cp.asnumpy(self.registers[idx, :8]).tolist()
            
            # Get partial sorting details
            length = int(self.sort_lengths[idx])
            start_pos = int(self.sort_start_positions[idx])
            direction = 'descending' if int(self.sort_directions[idx]) else 'ascending'
            
            # Extract the actual sorted sequence
            sorted_sequence = final_registers[start_pos:start_pos + length]
            
            discovery = {
                'instance_id': idx,
                'sort_cycle': int(self.sort_cycles[idx]),
                'partial_sorting': {
                    'length': length,
                    'start_position': start_pos,
                    'end_position': start_pos + length - 1,
                    'direction': direction,
                    'sequence': sorted_sequence,
                    'sequence_range': f"V{start_pos}-V{start_pos + length - 1}"
                },
                'initial_registers': [8, 3, 6, 1, 7, 2, 5, 4],  # We know what we put there
                'final_registers': final_registers,
                'register_activity': {
                    'total_register_ops': int(self.total_register_ops[idx]),
                    'register_reads': int(self.register_reads[idx]),
                    'register_writes': int(self.register_writes[idx])
                },
                'rom_data': cp.asnumpy(self.memory[idx, PROGRAM_START:]).tobytes(),
                'detection_info': {
                    'method': 'partial_consecutive_sorting',
                    'minimum_length': MIN_PARTIAL_LENGTH,
                    'enhancement_type': 'incremental_progress_detection'
                }
            }
            
            discoveries.append(discovery)
        
        return discoveries
    
    def reset(self):
        """Reset all state for next batch - optimized"""
        # Reset only what's necessary - avoid unnecessary memory operations
        self.registers.fill(0)
        self.index_register.fill(0)
        self.program_counter.fill(PROGRAM_START)
        self.stack_pointer.fill(0)
        self.delay_timer.fill(0)
        self.sound_timer.fill(0)
        
        # Reset state flags
        self.crashed.fill(0)
        self.halted.fill(0)
        self.waiting_for_key.fill(0)
        self.key_register.fill(0)
        
        # Reset partial sorting detection
        self.sort_cycles.fill(0)
        self.sort_achieved.fill(0)
        self.sort_lengths.fill(0)
        self.sort_start_positions.fill(0)
        self.sort_directions.fill(0)
        
        # Reset register counters
        self.total_register_ops.fill(0)
        self.register_reads.fill(0)
        self.register_writes.fill(0)
        
        # Reset RNG - use faster method
        self.rng_state = cp.random.randint(1, 2**32, size=self.num_instances, dtype=cp.uint32)
        
        # Reload font - optimized batch operation
        font_data = cp.tile(CHIP8_FONT, (self.num_instances, 1))
        self.memory[:, FONT_START:FONT_START + len(CHIP8_FONT)] = font_data


def generate_pure_random_roms_gpu(num_roms: int, rom_size: int = 3584) -> cp.ndarray:
    """Generate completely random ROMs on GPU - optimized for speed"""
    
    start_time = time.time()
    
    # Generate random data using cupy's optimized RNG
    all_random_data_gpu = cp.random.randint(0, 256, size=(num_roms, rom_size), dtype=cp.uint8)
    
    generation_time = time.time() - start_time
    
    # Fix division by zero - ensure minimum time
    generation_time = max(generation_time, 1e-6)  # Minimum 1 microsecond
    roms_per_second = num_roms / generation_time
    
    return all_random_data_gpu


def generate_pure_random_roms(num_roms: int, rom_size: int = 3584) -> List[np.ndarray]:
    """Generate completely random ROMs - legacy interface for compatibility"""
    print(f"Generating {num_roms:,} pure random ROMs ({rom_size} bytes each)...")
    
    start_time = time.time()
    
    try:
        # Generate on GPU first
        all_random_data_gpu = generate_pure_random_roms_gpu(num_roms, rom_size)
        
        # Copy to CPU in one transfer
        all_random_data = cp.asnumpy(all_random_data_gpu)
        
        # Clean up GPU memory
        del all_random_data_gpu
        
    except Exception as e:
        print(f"   GPU generation failed ({e}), falling back to CPU...")
        all_random_data = np.random.randint(0, 256, size=(num_roms, rom_size), dtype=np.uint8)
    
    # Convert to list - this is the slow part!
    rom_list = [all_random_data[i] for i in range(num_roms)]
    
    generation_time = time.time() - start_time
    roms_per_second = num_roms / generation_time
    
    print(f"   Total time: {generation_time:.3f}s ({roms_per_second:,.0f} ROMs/sec)")
    
    return rom_list


def save_partial_sorting_discovery_rom(discovery: Dict, output_dir: Path, batch_num: int, discovery_num: int) -> str:
    """Save a discovered ROM with partial sorting metadata (only for length > 5)"""
    
    # Check if this discovery meets the minimum save length requirement
    length = discovery['partial_sorting']['length']
    if length < MIN_SAVE_LENGTH:
        # Don't save short sequences, just return empty string
        return ""
    
    # Extract ROM data
    rom_data = discovery['rom_data']
    
    # Find actual end of ROM (trim trailing zeros)
    rom_array = np.frombuffer(rom_data, dtype=np.uint8)
    rom_end = len(rom_array)
    
    zero_count = 0
    for i in range(len(rom_array)):
        if rom_array[i] == 0:
            zero_count += 1
            if zero_count > 64:  # Long stretch of zeros = end of actual code
                rom_end = max(100, i - 63)  # Keep minimum size
                break
        else:
            zero_count = 0
    
    actual_rom = rom_array[:rom_end]
    
    # Generate hash for filename
    rom_hash = hashlib.sha256(actual_rom.tobytes()).hexdigest()[:8]
    
    # Create partial sorting filename with details
    cycle = discovery['sort_cycle']
    partial = discovery['partial_sorting']
    direction = partial['direction'][:3].upper()  # ASC or DESC
    sequence_range = partial['sequence_range']
    
    filename = f"LONGPARTIAL_B{batch_num:04d}D{discovery_num:02d}_{sequence_range}_L{length}_{direction}_C{cycle}_{rom_hash}"
    
    # Save ROM binary
    rom_path = output_dir / f"{filename}.ch8"
    with open(rom_path, 'wb') as f:
        f.write(actual_rom.tobytes())
    
    # Save partial sorting metadata
    metadata = {
        'filename': f"{filename}.ch8",
        'discovery_info': {
            'batch': batch_num,
            'discovery_number': discovery_num,
            'instance_id': discovery['instance_id'],
            'sort_cycle': discovery['sort_cycle'],
            'timestamp': time.time(),
            'discovery_type': 'long_partial_consecutive_sorting',
            'minimum_save_length': MIN_SAVE_LENGTH
        },
        'partial_sorting': discovery['partial_sorting'],
        'registers': {
            'initial': discovery['initial_registers'],
            'final': discovery['final_registers']
        },
        'register_activity': discovery['register_activity'],
        'detection_enhancement': discovery['detection_info'],
        'rom_info': {
            'size_bytes': len(actual_rom),
            'sha256_hash': rom_hash
        }
    }
    
    metadata_path = output_dir / f"{filename}.json"
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    return filename


def test_partial_sorting_babelscope():
    """Test the optimized partial sorting Babelscope implementation"""
    print("Testing Optimized Partial Sorting Babelscope Implementation")
    print("=" * 60)
    
    # Small test first
    print("Running small test (1000 ROMs)...")
    detector = PartialSortingBabelscopeDetector(1000)
    
    # Generate test ROMs
    test_roms = generate_pure_random_roms_gpu(100)
    detector.load_random_roms_and_setup_register_test(test_roms)
    
    # Run short test
    sorts_found = detector.run_partial_sorting_search(cycles=10000, check_interval=100)
    
    print(f"Test complete: {sorts_found} partial sorts found")
    
    if sorts_found > 0:
        discoveries = detector.get_partial_sorting_discoveries()
        print("Sample discovery:")
        discovery = discoveries[0]
        print(f"  Initial: {discovery['initial_registers']}")
        print(f"  Final: {discovery['final_registers']}")
        print(f"  Partial sort: {discovery['partial_sorting']['sequence_range']}")
        print(f"  Length: {discovery['partial_sorting']['length']}")
        print(f"  Direction: {discovery['partial_sorting']['direction']}")
        print(f"  Sequence: {discovery['partial_sorting']['sequence']}")
        print(f"  Cycle: {discovery['sort_cycle']}")
    
    return sorts_found > 0


if __name__ == "__main__":
    print("OPTIMIZED PARTIAL SORTING BABELSCOPE")
    print("=" * 60)
    print("Complete CHIP-8 emulation + partial sorting detection")
    print(f"Detecting {MIN_PARTIAL_LENGTH}+ consecutive sorted elements in V0-V7")
    print("Captures incremental progress toward full sorting")
    print("PERFORMANCE OPTIMIZATIONS: Register caching, early termination, reduced overhead")
    print()
    
    # Test basic functionality
    if test_partial_sorting_babelscope():
        print("Optimized Partial Sorting Babelscope working correctly!")
    else:
        print("No partial sorting found in test run")
    
    # Example usage
    print("\nExample usage:")
    print("detector = PartialSortingBabelscopeDetector(batch_size=100000)")
    print("detector.run_partial_sorting_search(cycles_per_rom=50000)")
    print()
    print("Enhancement: Detects 3+ consecutive sorted elements")
    print("Examples: [1,2,3], [8,7,6], [1,2,3,4], [8,7,6,5], etc.")
    print("Expected: Much higher discovery rate than full sorting!")
    print("Optimizations: Faster execution, cleaner output, reduced memory overhead")