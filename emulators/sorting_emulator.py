"""
Enhanced Babelscope Implementation: Multi-location Sorting Detection
Now searches for sorting algorithms across multiple 8-byte chunks in memory range 0x300-0xF000
This dramatically increases the discovery probability by ~480x!
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

# Multi-location sort test constants
SORT_ARRAY_SIZE = 8
SORT_SEARCH_START = 0x300
SORT_SEARCH_END = 0xF00  # Leave some room at the end
# Calculate number of 8-byte chunks we can fit
SORT_LOCATIONS_COUNT = (SORT_SEARCH_END - SORT_SEARCH_START) // SORT_ARRAY_SIZE

print(f"🎯 Multi-location setup: {SORT_LOCATIONS_COUNT} locations from 0x{SORT_SEARCH_START:03X} to 0x{SORT_SEARCH_END:03X}")

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

# Enhanced CHIP-8 emulation kernel with multi-location sort detection
ENHANCED_CHIP8_KERNEL = r'''
extern "C" __global__ __launch_bounds__(256, 4)
void chip8_multilocation_kernel(
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
    
    // Multi-location sort detection arrays
    unsigned int* __restrict__ sort_cycles,          // [instances] - cycle when sorted
    unsigned char* __restrict__ sort_achieved,       // [instances] - 1 if sorted
    unsigned short* __restrict__ sort_locations,     // [instances] - which location got sorted
    unsigned char* __restrict__ sort_directions,     // [instances] - 0=ascending, 1=descending
    
    // Memory access tracking
    unsigned int* __restrict__ total_reads,          // [instances]
    unsigned int* __restrict__ total_writes,         // [instances]
    unsigned int* __restrict__ sort_area_reads,      // [instances]
    unsigned int* __restrict__ sort_area_writes,     // [instances]
    
    // Execution parameters
    int num_instances,
    int cycles_to_run,
    int sort_check_interval,
    int sort_search_start,
    int sort_search_end,
    int sort_array_size,
    
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
    unsigned int local_total_reads = 0;
    unsigned int local_total_writes = 0;
    unsigned int local_sort_area_reads = 0;
    unsigned int local_sort_area_writes = 0;
    
    // Early exit if crashed or already sorted
    if (crashed[instance] || sort_achieved[instance]) {
        return;
    }
    
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
                if (registers[reg_base + x] == kk) {
                    pc += 2;
                }
                break;
                
            case 0x4:
                // SNE Vx, byte - Skip if Vx != byte
                if (registers[reg_base + x] != kk) {
                    pc += 2;
                }
                break;
                
            case 0x5:
                // SE Vx, Vy - Skip if Vx == Vy
                if (n == 0 && registers[reg_base + x] == registers[reg_base + y]) {
                    pc += 2;
                }
                break;
                
            case 0x6:
                // LD Vx, byte - Load byte into Vx
                registers[reg_base + x] = kk;
                break;
                
            case 0x7:
                // ADD Vx, byte - Add byte to Vx
                registers[reg_base + x] = (registers[reg_base + x] + kk) & 0xFF;
                break;
                
            case 0x8:
                // Register operations
                {
                    const unsigned char vx = registers[reg_base + x];
                    const unsigned char vy = registers[reg_base + y];
                    
                    switch (n) {
                        case 0x0: // LD Vx, Vy
                            registers[reg_base + x] = vy;
                            break;
                        case 0x1: // OR Vx, Vy
                            registers[reg_base + x] = vx | vy;
                            registers[reg_base + 0xF] = 0;
                            break;
                        case 0x2: // AND Vx, Vy
                            registers[reg_base + x] = vx & vy;
                            registers[reg_base + 0xF] = 0;
                            break;
                        case 0x3: // XOR Vx, Vy
                            registers[reg_base + x] = vx ^ vy;
                            registers[reg_base + 0xF] = 0;
                            break;
                        case 0x4: // ADD Vx, Vy
                            {
                                const int result = vx + vy;
                                registers[reg_base + x] = result & 0xFF;
                                registers[reg_base + 0xF] = (result > 255) ? 1 : 0;
                            }
                            break;
                        case 0x5: // SUB Vx, Vy
                            registers[reg_base + x] = (vx - vy) & 0xFF;
                            registers[reg_base + 0xF] = (vx >= vy) ? 1 : 0;
                            break;
                        case 0x6: // SHR Vx
                            registers[reg_base + x] = vx >> 1;
                            registers[reg_base + 0xF] = vx & 0x1;
                            break;
                        case 0x7: // SUBN Vx, Vy
                            registers[reg_base + x] = (vy - vx) & 0xFF;
                            registers[reg_base + 0xF] = (vy >= vx) ? 1 : 0;
                            break;
                        case 0xE: // SHL Vx
                            registers[reg_base + x] = (vx << 1) & 0xFF;
                            registers[reg_base + 0xF] = (vx & 0x80) ? 1 : 0;
                            break;
                        default:
                            crashed[instance] = 1;
                            break;
                    }
                }
                break;
                
            case 0x9:
                // SNE Vx, Vy - Skip if Vx != Vy
                if (n == 0 && registers[reg_base + x] != registers[reg_base + y]) {
                    pc += 2;
                }
                break;
                
            case 0xA:
                // LD I, addr - Load address into I
                index_reg = nnn;
                break;
                
            case 0xB:
                // JP V0, addr - Jump to V0 + addr
                pc = nnn + registers[reg_base + 0];
                break;
                
            case 0xC:
                // RND Vx, byte - Random number AND byte
                {
                    rng_state[instance] = rng_state[instance] * 1664525 + 1013904223;
                    const unsigned char rand_byte = (rng_state[instance] >> 16) & 0xFF;
                    registers[reg_base + x] = rand_byte & kk;
                }
                break;
                
            case 0xD:
                // DRW Vx, Vy, nibble - Draw sprite
                {
                    const unsigned char start_x = registers[reg_base + x] % 64;
                    const unsigned char start_y = registers[reg_base + y] % 32;
                    registers[reg_base + 0xF] = 0;
                    
                    for (int row = 0; row < n; row++) {
                        if (start_y + row >= 32) break;
                        if (index_reg + row >= 4096) break;
                        
                        const unsigned char sprite_byte = memory[mem_base + index_reg + row];
                        
                        for (int col = 0; col < 8; col++) {
                            if (start_x + col >= 64) break;
                            
                            if (sprite_byte & (0x80 >> col)) {
                                const int pixel_index = display_base + (start_y + row) * 64 + (start_x + col);
                                
                                if (display[pixel_index]) {
                                    registers[reg_base + 0xF] = 1;
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
                        break;
                    case 0x0A: // LD Vx, K - Wait for key
                        waiting_for_key[instance] = 1;
                        key_registers[instance] = x;
                        break;
                    case 0x15: // LD DT, Vx
                        dt = registers[reg_base + x];
                        break;
                    case 0x18: // LD ST, Vx
                        st = registers[reg_base + x];
                        break;
                    case 0x1E: // ADD I, Vx
                        index_reg = (index_reg + registers[reg_base + x]) & 0xFFFF;
                        break;
                    case 0x29: // LD F, Vx - Set I to font location
                        {
                            const unsigned char digit = registers[reg_base + x] & 0xF;
                            index_reg = 0x50 + digit * 5;
                        }
                        break;
                    case 0x33: // LD B, Vx - Store BCD
                        {
                            const unsigned char value = registers[reg_base + x];
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
                                local_total_writes++;
                                
                                // Track sort area writes
                                if (index_reg + i >= sort_search_start && index_reg + i < sort_search_end) {
                                    local_sort_area_writes++;
                                }
                            }
                        }
                        index_reg = (index_reg + x + 1) & 0xFFFF;
                        break;
                    case 0x65: // LD Vx, [I] - Load registers
                        for (int i = 0; i <= x; i++) {
                            if (index_reg + i < 4096) {
                                registers[reg_base + i] = memory[mem_base + index_reg + i];
                                local_total_reads++;
                                
                                // Track sort area reads
                                if (index_reg + i >= sort_search_start && index_reg + i < sort_search_end) {
                                    local_sort_area_reads++;
                                }
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
        
        // Check for sorting at multiple locations every N cycles
        if ((cycle % sort_check_interval) == 0 && !sort_achieved[instance]) {
            // Check all possible 8-byte chunks in the search range
            for (int offset = sort_search_start; offset <= sort_search_end - sort_array_size; offset += sort_array_size) {
                bool is_ascending = true;
                bool is_descending = true;
                
                // Check if this 8-byte chunk is sorted
                for (int i = 0; i < sort_array_size; i++) {
                    const unsigned char value = memory[mem_base + offset + i];
                    
                    // Check ascending: [1,2,3,4,5,6,7,8]
                    if (value != (i + 1)) {
                        is_ascending = false;
                    }
                    
                    // Check descending: [8,7,6,5,4,3,2,1]
                    if (value != (sort_array_size - i)) {
                        is_descending = false;
                    }
                }
                
                if (is_ascending || is_descending) {
                    sort_achieved[instance] = 1;
                    sort_cycles[instance] = cycle;
                    sort_locations[instance] = offset;
                    sort_directions[instance] = is_descending ? 1 : 0;
                    break; // Found one! Early termination
                }
            }
            
            if (sort_achieved[instance]) {
                break; // Exit main loop early
            }
        }
        
        // Update timers periodically
        if ((cycle & 15) == 0) {
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
    total_reads[instance] += local_total_reads;
    total_writes[instance] += local_total_writes;
    sort_area_reads[instance] += local_sort_area_reads;
    sort_area_writes[instance] += local_sort_area_writes;
}
'''

class EnhancedBabelscopeDetector:
    """
    Enhanced Babelscope implementation: Multi-location sorting detection
    Dramatically increases discovery probability by monitoring many locations
    """
    
    def __init__(self, num_instances: int):
        print(f"🔬 Initializing Enhanced Multi-Location Babelscope Detector")
        print(f"   Instances: {num_instances:,}")
        print(f"   Monitoring {SORT_LOCATIONS_COUNT} locations (8-byte chunks)")
        print(f"   Search range: 0x{SORT_SEARCH_START:03X} to 0x{SORT_SEARCH_END:03X}")
        print(f"   Effective discovery area increased by ~{SORT_LOCATIONS_COUNT}x!")
        
        self.num_instances = num_instances
        
        # Calculate grid configuration
        self.block_size = 256
        self.grid_size = (num_instances + self.block_size - 1) // self.block_size
        
        print(f"   Block size: {self.block_size}")
        print(f"   Grid size: {self.grid_size}")
        
        # Compile the enhanced CHIP-8 kernel
        print("   Compiling enhanced multi-location CHIP-8 kernel...")
        try:
            device = cp.cuda.Device()
            device_props = cp.cuda.runtime.getDeviceProperties(device.id)
            compute_capability = device.compute_capability
            
            compile_options = [
                '--use_fast_math',
                '--opt-level=3',
                f'--gpu-architecture=sm_{compute_capability[0]}{compute_capability[1]}'
            ]
            
            self.kernel = cp.RawKernel(
                ENHANCED_CHIP8_KERNEL, 
                'chip8_multilocation_kernel',
                options=compile_options
            )
        except Exception as e:
            print(f"   ⚠️  Compiling without GPU-specific optimizations: {e}")
            self.kernel = cp.RawKernel(ENHANCED_CHIP8_KERNEL, 'chip8_multilocation_kernel')
        
        # Initialize state
        self._initialize_state()
        
        print("✅ Enhanced Multi-Location Babelscope ready!")
    
    def _initialize_state(self):
        """Initialize all GPU arrays"""
        print("   Allocating GPU memory...")
        
        # Core CHIP-8 state
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
        
        # Enhanced sort detection - now tracks which location and direction
        self.sort_cycles = cp.zeros(self.num_instances, dtype=cp.uint32)
        self.sort_achieved = cp.zeros(self.num_instances, dtype=cp.uint8)
        self.sort_locations = cp.zeros(self.num_instances, dtype=cp.uint16)  # Which address got sorted
        self.sort_directions = cp.zeros(self.num_instances, dtype=cp.uint8)  # 0=ascending, 1=descending
        
        # Enhanced memory access tracking
        self.total_reads = cp.zeros(self.num_instances, dtype=cp.uint32)
        self.total_writes = cp.zeros(self.num_instances, dtype=cp.uint32)
        self.sort_area_reads = cp.zeros(self.num_instances, dtype=cp.uint32)
        self.sort_area_writes = cp.zeros(self.num_instances, dtype=cp.uint32)
        
        # RNG state
        self.rng_state = cp.random.randint(1, 2**32, size=self.num_instances, dtype=cp.uint32)
        
        # Load font into all instances
        font_data = cp.tile(CHIP8_FONT, (self.num_instances, 1))
        self.memory[:, FONT_START:FONT_START + len(CHIP8_FONT)] = font_data
        
        memory_usage = (
            self.memory.nbytes + self.display.nbytes + self.registers.nbytes +
            self.sort_cycles.nbytes + self.sort_achieved.nbytes + 
            self.sort_locations.nbytes + self.sort_directions.nbytes +
            self.total_reads.nbytes + self.total_writes.nbytes +
            self.sort_area_reads.nbytes + self.sort_area_writes.nbytes
        ) / (1024**3)
        
        print(f"   Memory allocated: {memory_usage:.2f} GB")
    
    def load_random_roms_and_setup_multilocation_test(self, rom_data):
        """Load random ROMs and setup multi-location sort test"""
        
        if isinstance(rom_data, cp.ndarray):
            print(f"📥 Loading {rom_data.shape[0]:,} random ROMs from GPU array...")
            
            num_roms_to_load = min(rom_data.shape[0], self.num_instances)
            rom_size = min(rom_data.shape[1], MEMORY_SIZE - PROGRAM_START)
            rom_end = PROGRAM_START + rom_size
            
            self.memory[:num_roms_to_load, PROGRAM_START:rom_end] = rom_data[:num_roms_to_load, :rom_size]
            
            if num_roms_to_load < self.num_instances:
                for i in range(num_roms_to_load, self.num_instances):
                    rom_idx = i % num_roms_to_load
                    self.memory[i, PROGRAM_START:rom_end] = rom_data[rom_idx, :rom_size]
        else:
            print(f"📥 Loading {len(rom_data):,} random ROMs from CPU list...")
            
            for i in range(self.num_instances):
                rom_array = rom_data[i % len(rom_data)]
                
                if len(rom_array) > MEMORY_SIZE - PROGRAM_START:
                    rom_array = rom_array[:MEMORY_SIZE - PROGRAM_START]
                
                rom_end = PROGRAM_START + len(rom_array)
                self.memory[i, PROGRAM_START:rom_end] = cp.array(rom_array)
        
        # Setup the test pattern at MULTIPLE locations
        test_pattern = np.array([8, 3, 6, 1, 7, 2, 5, 4], dtype=np.uint8)
        
        print(f"🎯 Setting up multi-location test:")
        print(f"   Test pattern: {test_pattern}")
        print(f"   Locations: {SORT_LOCATIONS_COUNT} chunks from 0x{SORT_SEARCH_START:03X} to 0x{SORT_SEARCH_END:03X}")
        
        # Place test pattern at ALL possible 8-byte aligned locations
        test_pattern_gpu = cp.array(test_pattern)
        
        location_count = 0
        for offset in range(SORT_SEARCH_START, SORT_SEARCH_END, SORT_ARRAY_SIZE):
            if offset + SORT_ARRAY_SIZE <= SORT_SEARCH_END:
                self.memory[:, offset:offset + SORT_ARRAY_SIZE] = test_pattern_gpu[None, :]
                location_count += 1
        
        print(f"   ✅ Placed test pattern at {location_count} locations")
        print(f"   🎯 Discovery probability increased by ~{location_count}x!")
    
    def run_enhanced_babelscope_search(self, cycles: int = 100000, check_interval: int = 100) -> int:
        """Run the enhanced multi-location Babelscope search"""
        print(f"🔍 Running Enhanced Multi-Location Babelscope search:")
        print(f"   Cycles: {cycles:,}, Check interval: {check_interval}")
        print(f"   Monitoring {SORT_LOCATIONS_COUNT} locations per ROM")
        print(f"   Effective search rate: {self.num_instances * SORT_LOCATIONS_COUNT:,} location-checks")
        
        start_time = time.time()
        
        # Launch the enhanced CHIP-8 kernel
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
                
                # Enhanced sort detection
                self.sort_cycles,
                self.sort_achieved,
                self.sort_locations,
                self.sort_directions,
                
                # Enhanced memory access tracking
                self.total_reads,
                self.total_writes,
                self.sort_area_reads,
                self.sort_area_writes,
                
                # Parameters
                self.num_instances,
                cycles,
                check_interval,
                SORT_SEARCH_START,
                SORT_SEARCH_END,
                SORT_ARRAY_SIZE,
                
                # RNG
                self.rng_state
            )
        )
        
        # Synchronize and get results
        cp.cuda.Stream.null.synchronize()
        execution_time = time.time() - start_time
        
        # Count results
        sorts_found = int(cp.sum(self.sort_achieved))
        crashed_count = int(cp.sum(self.crashed))
        sort_area_accessed = int(cp.sum((self.sort_area_reads > 0) | (self.sort_area_writes > 0)))
        
        # Performance metrics
        roms_per_second = self.num_instances / execution_time
        effective_checks_per_second = (self.num_instances * SORT_LOCATIONS_COUNT) / execution_time
        
        print(f"⚡ Execution time: {execution_time:.3f}s")
        print(f"⚡ {roms_per_second:,.0f} ROMs/sec")
        print(f"⚡ {effective_checks_per_second:,.0f} location-checks/sec")
        print(f"🎯 Sorting algorithms found: {sorts_found}")
        print(f"💥 Crashed instances: {crashed_count}")
        print(f"📊 Instances that accessed sort area: {sort_area_accessed}")
        
        if sorts_found > 0:
            print(f"🎉 SUCCESS! Found {sorts_found} sorting algorithm(s)!")
            
            # Show details of discoveries
            sorted_indices = cp.where(self.sort_achieved)[0]
            for idx in sorted_indices[:5]:  # Show first 5
                idx = int(idx)
                location = int(self.sort_locations[idx])
                direction = "descending" if self.sort_directions[idx] else "ascending"
                cycle = int(self.sort_cycles[idx])
                print(f"   Discovery {idx}: 0x{location:03X} -> {direction} at cycle {cycle}")
        
        return sorts_found
    
    def get_enhanced_discoveries(self) -> List[Dict]:
        """Get all discovered sorting algorithms with enhanced metadata"""
        discoveries = []
        
        # Find instances that achieved sorting
        sorted_indices = cp.where(self.sort_achieved)[0]
        
        for idx in sorted_indices:
            idx = int(idx)
            
            # Get the sorted location and extract the final array
            sort_location = int(self.sort_locations[idx])
            final_array = cp.asnumpy(self.memory[idx, sort_location:sort_location + SORT_ARRAY_SIZE]).tolist()
            
            discovery = {
                'instance_id': idx,
                'sort_cycle': int(self.sort_cycles[idx]),
                'sort_location': sort_location,
                'sort_direction': 'descending' if int(self.sort_directions[idx]) else 'ascending',
                'initial_array': [8, 3, 6, 1, 7, 2, 5, 4],  # We know what we put there
                'final_array': final_array,
                'memory_access': {
                    'total_reads': int(self.total_reads[idx]),
                    'total_writes': int(self.total_writes[idx]),
                    'sort_area_reads': int(self.sort_area_reads[idx]),
                    'sort_area_writes': int(self.sort_area_writes[idx])
                },
                'rom_data': cp.asnumpy(self.memory[idx, PROGRAM_START:]).tobytes(),
                'multilocation_info': {
                    'total_locations_monitored': SORT_LOCATIONS_COUNT,
                    'search_range': f"0x{SORT_SEARCH_START:03X}-0x{SORT_SEARCH_END:03X}",
                    'discovery_probability_multiplier': SORT_LOCATIONS_COUNT
                }
            }
            
            discoveries.append(discovery)
        
        return discoveries
    
    def reset(self):
        """Reset all state for next batch"""
        self.registers.fill(0)
        self.index_register.fill(0)
        self.program_counter.fill(PROGRAM_START)
        self.stack_pointer.fill(0)
        self.stack.fill(0)
        self.delay_timer.fill(0)
        self.sound_timer.fill(0)
        self.keypad.fill(0)
        
        # Reset state flags
        self.crashed.fill(0)
        self.halted.fill(0)
        self.waiting_for_key.fill(0)
        self.key_register.fill(0)
        
        # Reset enhanced sort detection
        self.sort_cycles.fill(0)
        self.sort_achieved.fill(0)
        self.sort_locations.fill(0)
        self.sort_directions.fill(0)
        
        # Reset enhanced counters
        self.total_reads.fill(0)
        self.total_writes.fill(0)
        self.sort_area_reads.fill(0)
        self.sort_area_writes.fill(0)
        
        # Reset RNG
        self.rng_state = cp.random.randint(1, 2**32, size=self.num_instances, dtype=cp.uint32)
        
        # Reload font
        font_data = cp.tile(CHIP8_FONT, (self.num_instances, 1))
        self.memory[:, FONT_START:FONT_START + len(CHIP8_FONT)] = font_data


def generate_pure_random_roms_gpu(num_roms: int, rom_size: int = 3584) -> cp.ndarray:
    """Generate completely random ROMs on GPU - returns GPU array for efficiency"""
    print(f"🎲 Generating {num_roms:,} pure random ROMs on GPU ({rom_size} bytes each)...")
    
    start_time = time.time()
    
    # Generate all random data on GPU and keep it there!
    all_random_data_gpu = cp.random.randint(0, 256, size=(num_roms, rom_size), dtype=cp.uint8)
    
    generation_time = time.time() - start_time
    roms_per_second = num_roms / generation_time
    
    print(f"   ✅ Generated on GPU in {generation_time:.3f}s ({roms_per_second:,.0f} ROMs/sec)")
    
    return all_random_data_gpu


def generate_pure_random_roms(num_roms: int, rom_size: int = 3584) -> List[np.ndarray]:
    """Generate completely random ROMs - legacy interface for compatibility"""
    print(f"🎲 Generating {num_roms:,} pure random ROMs ({rom_size} bytes each)...")
    
    start_time = time.time()
    
    try:
        # Generate on GPU first
        all_random_data_gpu = generate_pure_random_roms_gpu(num_roms, rom_size)
        
        # Copy to CPU in one transfer
        all_random_data = cp.asnumpy(all_random_data_gpu)
        
        # Clean up GPU memory
        del all_random_data_gpu
        
    except Exception as e:
        print(f"   ⚠️  GPU generation failed ({e}), falling back to CPU...")
        all_random_data = np.random.randint(0, 256, size=(num_roms, rom_size), dtype=np.uint8)
    
    # Convert to list - this is the slow part!
    rom_list = [all_random_data[i] for i in range(num_roms)]
    
    generation_time = time.time() - start_time
    roms_per_second = num_roms / generation_time
    
    print(f"   ✅ Total time: {generation_time:.3f}s ({roms_per_second:,.0f} ROMs/sec)")
    
    return rom_list


def save_enhanced_discovery_rom(discovery: Dict, output_dir: Path, batch_num: int, discovery_num: int) -> str:
    """Save a discovered ROM with enhanced metadata"""
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
    
    # Create enhanced filename with location info
    cycle = discovery['sort_cycle']
    location = discovery['sort_location']
    direction = discovery['sort_direction'][:3].upper()  # ASC or DESC
    filename = f"MULTILOC_B{batch_num:04d}D{discovery_num:02d}_0x{location:03X}_{direction}_C{cycle}_{rom_hash}"
    
    # Save ROM binary
    rom_path = output_dir / f"{filename}.ch8"
    with open(rom_path, 'wb') as f:
        f.write(actual_rom.tobytes())
    
    # Save enhanced metadata
    metadata = {
        'filename': f"{filename}.ch8",
        'discovery_info': {
            'batch': batch_num,
            'discovery_number': discovery_num,
            'instance_id': discovery['instance_id'],
            'sort_cycle': discovery['sort_cycle'],
            'sort_location': f"0x{discovery['sort_location']:03X}",
            'sort_direction': discovery['sort_direction'],
            'timestamp': time.time(),
            'discovery_type': 'multi_location_enhanced'
        },
        'arrays': {
            'initial': discovery['initial_array'],
            'final': discovery['final_array']
        },
        'memory_access': discovery['memory_access'],
        'multilocation_enhancement': discovery['multilocation_info'],
        'rom_info': {
            'size_bytes': len(actual_rom),
            'sha256_hash': rom_hash
        }
    }
    
    metadata_path = output_dir / f"{filename}.json"
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"   💾 Saved: {filename}.ch8 ({len(actual_rom)} bytes)")
    print(f"      Location: 0x{discovery['sort_location']:03X}, Direction: {discovery['sort_direction']}")
    
    return filename


class EnhancedBabelscopeRunner:
    """Enhanced runner for the multi-location Babelscope approach"""
    
    def __init__(self, batch_size: int = 50000, output_dir: str = "enhanced_babelscope_output"):
        self.batch_size = batch_size
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Session tracking
        self.session_stats = {
            'total_roms_tested': 0,
            'total_batches': 0,
            'total_discoveries': 0,
            'effective_location_checks': 0,
            'start_time': time.time(),
            'enhancement_type': 'multi_location',
            'locations_per_rom': SORT_LOCATIONS_COUNT
        }
        
        print(f"🔬 Enhanced Multi-Location Babelscope Runner")
        print(f"   Batch size: {batch_size:,}")
        print(f"   Locations per ROM: {SORT_LOCATIONS_COUNT}")
        print(f"   Effective discovery rate multiplier: ~{SORT_LOCATIONS_COUNT}x")
        print(f"   Output: {output_dir}")
        
        # Initialize enhanced detector
        self.detector = EnhancedBabelscopeDetector(batch_size)
    
    def run_enhanced_search(self, num_batches: int = 10, cycles_per_rom: int = 100000):
        """Run the enhanced multi-location Babelscope search"""
        print(f"🏹 Starting Enhanced Multi-Location Babelscope search: {num_batches} batches")
        print(f"   Cycles per ROM: {cycles_per_rom:,}")
        print(f"   Expected discovery rate: ~{SORT_LOCATIONS_COUNT}x better than single-location")
        print()
        
        total_discoveries = 0
        
        for batch_num in range(1, num_batches + 1):
            print(f"🎯 BATCH {batch_num}/{num_batches}")
            print("-" * 40)
            
            # Generate pure random ROMs
            rom_data = generate_pure_random_roms_gpu(self.batch_size)
            
            # Load ROMs and setup multi-location sort test
            self.detector.load_random_roms_and_setup_multilocation_test(rom_data)
            
            # Run the enhanced search
            sorts_found = self.detector.run_enhanced_babelscope_search(cycles_per_rom, check_interval=100)
            
            # Process discoveries
            if sorts_found > 0:
                discoveries = self.detector.get_enhanced_discoveries()
                
                print(f"🎉 Found {len(discoveries)} sorting algorithms!")
                
                for i, discovery in enumerate(discoveries):
                    filename = save_enhanced_discovery_rom(discovery, self.output_dir, batch_num, i + 1)
                    total_discoveries += 1
                    
                    location = discovery['sort_location']
                    direction = discovery['sort_direction']
                    cycle = discovery['sort_cycle']
                    reads = discovery['memory_access']['sort_area_reads']
                    writes = discovery['memory_access']['sort_area_writes']
                    
                    print(f"      Discovery {i+1}: 0x{location:03X} {direction} @ cycle {cycle:,} ({reads}R/{writes}W)")
            
            # Update session stats
            self.session_stats['total_roms_tested'] += self.batch_size
            self.session_stats['total_batches'] = batch_num
            self.session_stats['total_discoveries'] = total_discoveries
            self.session_stats['effective_location_checks'] += self.batch_size * SORT_LOCATIONS_COUNT
            
            # Print session summary
            total_time = time.time() - self.session_stats['start_time']
            total_rate = self.session_stats['total_roms_tested'] / total_time
            effective_rate = self.session_stats['effective_location_checks'] / total_time
            
            print(f"📊 Session totals:")
            print(f"   ROMs tested: {self.session_stats['total_roms_tested']:,}")
            print(f"   Location-checks: {self.session_stats['effective_location_checks']:,}")
            print(f"   Discoveries: {total_discoveries}")
            print(f"   ROM rate: {total_rate:,.0f} ROMs/sec")
            print(f"   Location-check rate: {effective_rate:,.0f} checks/sec")
            
            if total_discoveries > 0:
                discovery_rate = self.session_stats['total_roms_tested'] // total_discoveries
                effective_discovery_rate = self.session_stats['effective_location_checks'] // total_discoveries
                print(f"   Discovery rate: 1 in {discovery_rate:,} ROMs")
                print(f"   Effective discovery rate: 1 in {effective_discovery_rate:,} location-checks")
            
            print()
            
            # Reset for next batch
            self.detector.reset()
            
            # Memory cleanup
            del rom_data
            cp.get_default_memory_pool().free_all_blocks()
        
        print("🏁 Enhanced Babelscope search complete!")
        print(f"   Total discoveries: {total_discoveries}")
        print(f"   Enhancement factor: ~{SORT_LOCATIONS_COUNT}x detection area")
        print(f"   Results saved in: {self.output_dir}")
        
        return total_discoveries


def test_enhanced_babelscope():
    """Test the enhanced multi-location Babelscope implementation"""
    print("🧪 Testing Enhanced Multi-Location Babelscope Implementation")
    print("=" * 60)
    
    # Small test first
    print("Running small test (1000 ROMs)...")
    detector = EnhancedBabelscopeDetector(1000)
    
    # Generate test ROMs
    test_roms = generate_pure_random_roms_gpu(100)
    detector.load_random_roms_and_setup_multilocation_test(test_roms)
    
    # Run short test
    sorts_found = detector.run_enhanced_babelscope_search(cycles=10000, check_interval=100)
    
    print(f"✅ Test complete: {sorts_found} sorts found")
    
    if sorts_found > 0:
        discoveries = detector.get_enhanced_discoveries()
        print("Sample discovery:")
        discovery = discoveries[0]
        print(f"  Initial: {discovery['initial_array']}")
        print(f"  Final: {discovery['final_array']}")
        print(f"  Location: 0x{discovery['sort_location']:03X}")
        print(f"  Direction: {discovery['sort_direction']}")
        print(f"  Cycle: {discovery['sort_cycle']}")
        print(f"  Access: {discovery['memory_access']['sort_area_reads']}R/{discovery['memory_access']['sort_area_writes']}W")
    
    return sorts_found > 0


if __name__ == "__main__":
    print("🔬 ENHANCED MULTI-LOCATION BABELSCOPE")
    print("=" * 60)
    print("🎯 Complete CHIP-8 emulation + multi-location sort detection")
    print(f"📊 Monitoring {SORT_LOCATIONS_COUNT} locations per ROM (~{SORT_LOCATIONS_COUNT}x discovery rate)")
    print("🧬 Pure computational archaeology with enhanced detection area")
    print()
    
    # Test basic functionality
    if test_enhanced_babelscope():
        print("✅ Enhanced Multi-Location Babelscope working correctly!")
    else:
        print("⚠️  No discoveries in test run (still possible, but less likely now)")
    
    # Example usage
    print("\n🚀 Example usage:")
    print("runner = EnhancedBabelscopeRunner(batch_size=100000)")
    print("runner.run_enhanced_search(num_batches=100, cycles_per_rom=50000)")
    print()
    print(f"Expected discovery rate: ~{SORT_LOCATIONS_COUNT}x better than single location")
    print("RTX 5080 should process ~100K+ ROMs/sec with millions of location-checks/sec")