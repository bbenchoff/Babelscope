"""
GTX 1070-Optimized CHIP-8 Emulator for Memory-Based CA Detection
Focuses on memory patterns, ignores display operations
"""

import cupy as cp
import numpy as np
from typing import Dict, Tuple, Optional, List, Union
import time

# CHIP-8 System Constants
MEMORY_SIZE = 4096
DISPLAY_WIDTH = 64
DISPLAY_HEIGHT = 32
REGISTER_COUNT = 16
STACK_SIZE = 16
KEYPAD_SIZE = 16
PROGRAM_START = 0x200
FONT_START = 0x50

# CHIP-8 Font set
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

# GTX 1070-optimized Memory-CA detection kernel
MEMORY_CA_KERNEL_SOURCE = r'''
extern "C" __global__
void chip8_memory_ca_kernel(
    // State arrays
    unsigned char* memory,              // [instances][4096]
    unsigned char* registers,           // [instances][16]
    unsigned short* index_registers,    // [instances]
    unsigned short* program_counters,   // [instances]
    unsigned char* stack_pointers,      // [instances]
    unsigned short* stacks,             // [instances][16]
    unsigned char* delay_timers,        // [instances]
    unsigned char* sound_timers,        // [instances]
    
    // State flags
    unsigned char* crashed,             // [instances]
    unsigned char* halted,              // [instances]
    
    // Statistics arrays (minimal for memory-CA focus)
    unsigned int* instructions_executed,    // [instances]
    unsigned int* memory_operations,        // [instances]
    
    // Random number state
    unsigned int* rng_state,            // [instances]
    
    // Memory-CA detection arrays
    unsigned char* ca_detected,         // [instances] - boolean CA flag
    float* ca_likelihood,               // [instances] - CA likelihood score
    unsigned short* hot_loop_start,     // [instances] - start of hot loop
    unsigned short* hot_loop_end,       // [instances] - end of hot loop
    unsigned int* pc_frequency,         // [instances][256] - PC frequency buckets
    unsigned int* memory_write_frequency, // [instances][128] - Memory write pattern tracking
    
    // Execution parameters
    int num_instances,
    int cycles_to_run,
    int timer_update_interval,
    
    // Memory-CA detection parameters
    int ca_detection_interval,          // How often to check for CA patterns
    float ca_threshold,                 // Minimum CA likelihood to flag
    
    // Quirks
    int quirk_memory,
    int quirk_jumping,
    int quirk_logic
) {
    int instance = blockIdx.x * blockDim.x + threadIdx.x;
    if (instance >= num_instances) return;
    
    // Calculate base indices for this instance
    int mem_base = instance * 4096;
    int reg_base = instance * 16;
    int stack_base = instance * 16;
    int pc_freq_base = instance * 256;  // 256 PC frequency buckets per instance
    int mem_write_base = instance * 128; // 128 memory write buckets
    
    // Local state (registers for better performance)
    unsigned short pc = program_counters[instance];
    unsigned short index_reg = index_registers[instance];
    unsigned char sp = stack_pointers[instance];
    unsigned char dt = delay_timers[instance];
    unsigned char st = sound_timers[instance];
    
    // Enhanced CA detection state with focus on memory patterns
    unsigned int local_pc_frequency[256];
    unsigned int local_memory_writes[128];
    unsigned int instruction_counts[16];  // Track instruction types
    for (int i = 0; i < 256; i++) local_pc_frequency[i] = 0;
    for (int i = 0; i < 128; i++) local_memory_writes[i] = 0;
    for (int i = 0; i < 16; i++) instruction_counts[i] = 0;
    
    unsigned int last_ca_check = 0;
    unsigned int total_memory_ops = 0;
    unsigned int sequential_memory_accesses = 0;
    unsigned short last_memory_write_addr = 0xFFFF;
    
    // Statistics
    unsigned int local_instructions = 0;
    unsigned int local_memory_operations = 0;
    
    // Check if this instance is active
    if (crashed[instance] || halted[instance]) {
        return; // Skip crashed/halted instances
    }
    
    // Main execution loop with enhanced memory-CA detection
    for (int cycle = 0; cycle < cycles_to_run; cycle++) {
        // Check PC bounds
        if (pc >= 4096 - 1) {
            crashed[instance] = 1;
            break;
        }
        
        // Track PC frequency for CA detection
        if (pc >= 0x200) {
            unsigned char pc_bucket = (pc - 0x200) / 16;  // Map 0x200-0xFFF to 0-255
            if (pc_bucket < 256) {
                local_pc_frequency[pc_bucket]++;
            }
        }
        
        // Fetch instruction
        unsigned char high_byte = memory[mem_base + pc];
        unsigned char low_byte = memory[mem_base + pc + 1];
        unsigned short instruction = (high_byte << 8) | low_byte;
        
        // Track instruction types for better CA analysis
        unsigned char opcode = (instruction & 0xF000) >> 12;
        instruction_counts[opcode]++;
        
        // Increment PC
        pc += 2;
        local_instructions++;
        
        // Decode instruction
        unsigned char x = (instruction & 0x0F00) >> 8;
        unsigned char y = (instruction & 0x00F0) >> 4;
        unsigned char n = instruction & 0x000F;
        unsigned char kk = instruction & 0x00FF;
        unsigned short nnn = instruction & 0x0FFF;
        
        // Execute instruction with enhanced memory tracking
        switch (opcode) {
            case 0x0:
                if (instruction == 0x00EE) {
                    if (sp > 0) {
                        sp--;
                        pc = stacks[stack_base + sp];
                    } else {
                        crashed[instance] = 1;
                    }
                }
                // Skip display clear (0x00E0) - we don't care about display
                break;
                
            case 0x1: pc = nnn; break;
            case 0x2:
                if (sp < 16) {
                    stacks[stack_base + sp] = pc;
                    sp++;
                    pc = nnn;
                } else {
                    crashed[instance] = 1;
                }
                break;
            case 0x3: if (registers[reg_base + x] == kk) pc += 2; break;
            case 0x4: if (registers[reg_base + x] != kk) pc += 2; break;
            case 0x5: if (n == 0 && registers[reg_base + x] == registers[reg_base + y]) pc += 2; break;
            case 0x6: registers[reg_base + x] = kk; break;
            case 0x7: registers[reg_base + x] = (registers[reg_base + x] + kk) & 0xFF; break;
            
            case 0x8: // Register operations
                {
                    unsigned char vx = registers[reg_base + x];
                    unsigned char vy = registers[reg_base + y];
                    unsigned char result = 0;
                    unsigned char flag = 0;
                    
                    switch (n) {
                        case 0x0: result = vy; break;
                        case 0x1: result = vx | vy; if (quirk_logic) flag = 0; break;
                        case 0x2: result = vx & vy; if (quirk_logic) flag = 0; break;
                        case 0x3: result = vx ^ vy; if (quirk_logic) flag = 0; break;
                        case 0x4: {
                            int sum = vx + vy;
                            result = sum & 0xFF;
                            flag = (sum > 255) ? 1 : 0;
                        } break;
                        case 0x5: result = (vx - vy) & 0xFF; flag = (vx >= vy) ? 1 : 0; break;
                        case 0x6: result = vx >> 1; flag = vx & 0x1; break;
                        case 0x7: result = (vy - vx) & 0xFF; flag = (vy >= vx) ? 1 : 0; break;
                        case 0xE: result = (vx << 1) & 0xFF; flag = (vx & 0x80) ? 1 : 0; break;
                        default: crashed[instance] = 1; continue;
                    }
                    
                    registers[reg_base + x] = result;
                    if (n >= 0x1 && n <= 0xE && n != 0x0) {
                        registers[reg_base + 0xF] = flag;
                    }
                }
                break;
                
            case 0x9: if (n == 0 && registers[reg_base + x] != registers[reg_base + y]) pc += 2; break;
            case 0xA: index_reg = nnn; break;
            case 0xB:
                if (quirk_jumping) {
                    pc = nnn + registers[reg_base + ((nnn & 0xF00) >> 8)];
                } else {
                    pc = nnn + registers[reg_base + 0];
                }
                break;
            case 0xC:
                {
                    rng_state[instance] = rng_state[instance] * 1664525 + 1013904223;
                    unsigned char random_byte = (rng_state[instance] >> 16) & 0xFF;
                    registers[reg_base + x] = random_byte & kk;
                }
                break;
                
            // Skip display operations (0xD) - we don't care about display for memory-CA
            case 0xD: break;
            
            // Skip key operations (0xE) - not relevant for memory-CA
            case 0xE: break;
                
            case 0xF: // Timer and CRITICAL memory operations
                switch (kk) {
                    case 0x07: registers[reg_base + x] = dt; break;
                    case 0x15: dt = registers[reg_base + x]; break;
                    case 0x18: st = registers[reg_base + x]; break;
                    case 0x1E: 
                        index_reg = (index_reg + registers[reg_base + x]) & 0xFFFF; 
                        total_memory_ops++;
                        break;
                    case 0x29: {
                        unsigned char digit = registers[reg_base + x] & 0xF;
                        index_reg = 0x50 + digit * 5;
                    } break;
                    case 0x33: {
                        unsigned char value = registers[reg_base + x];
                        if (index_reg + 2 < 4096) {
                            memory[mem_base + index_reg] = value / 100;
                            memory[mem_base + index_reg + 1] = (value / 10) % 10;
                            memory[mem_base + index_reg + 2] = value % 10;
                            
                            // Track memory write pattern
                            unsigned char write_bucket = (index_reg - 0x200) / 32;
                            if (write_bucket < 128) {
                                local_memory_writes[write_bucket]++;
                            }
                            
                            // Check for sequential memory writes
                            if (last_memory_write_addr != 0xFFFF && 
                                index_reg == last_memory_write_addr + 3) {
                                sequential_memory_accesses++;
                            }
                            last_memory_write_addr = index_reg;
                        }
                        total_memory_ops++;
                        local_memory_operations++;
                    } break;
                    case 0x55: // CRITICAL: Memory store operation
                        for (int i = 0; i <= x; i++) {
                            if (index_reg + i < 4096) {
                                memory[mem_base + index_reg + i] = registers[reg_base + i];
                                
                                // Track memory write pattern
                                unsigned char write_bucket = ((index_reg + i) - 0x200) / 32;
                                if (write_bucket < 128) {
                                    local_memory_writes[write_bucket]++;
                                }
                            }
                        }
                        
                        // Check for sequential memory writes
                        if (last_memory_write_addr != 0xFFFF && 
                            index_reg <= last_memory_write_addr + 16) {
                            sequential_memory_accesses += (x + 1);
                        }
                        last_memory_write_addr = index_reg + x;
                        
                        if (quirk_memory) {
                            index_reg = (index_reg + x + 1) & 0xFFFF;
                        }
                        total_memory_ops++;
                        local_memory_operations++;
                        break;
                    case 0x65: // CRITICAL: Memory load operation
                        for (int i = 0; i <= x; i++) {
                            if (index_reg + i < 4096) {
                                registers[reg_base + i] = memory[mem_base + index_reg + i];
                            }
                        }
                        if (quirk_memory) {
                            index_reg = (index_reg + x + 1) & 0xFFFF;
                        }
                        total_memory_ops++;
                        local_memory_operations++;
                        break;
                    default: break; // Ignore unknown F instructions
                }
                break;
                
            default: crashed[instance] = 1; break;
        }
        
        // ENHANCED MEMORY-CA DETECTION LOGIC
        if (cycle > 2000 && ca_detection_interval > 0 && 
            (cycle - last_ca_check) >= ca_detection_interval) {
            last_ca_check = cycle;
            
            // Find most frequent PC bucket (hot loop detection)
            unsigned int max_frequency = 0;
            unsigned char hot_bucket = 0;
            unsigned int total_frequency = 0;
            
            for (int i = 0; i < 256; i++) {
                total_frequency += local_pc_frequency[i];
                if (local_pc_frequency[i] > max_frequency) {
                    max_frequency = local_pc_frequency[i];
                    hot_bucket = i;
                }
            }
            
            // Check for hot loop
            if (total_frequency > 0 && max_frequency > total_frequency / 6) {
                // Calculate actual PC range from bucket
                unsigned short hot_start = 0x200 + hot_bucket * 16;
                unsigned short hot_end = hot_start + 32;
                
                // MEMORY-FOCUSED CA SCORING
                float ca_score = 0.0f;
                
                // Pattern counters for memory-CA detection
                int add_i_count = 0;
                int memory_load_count = 0;
                int memory_store_count = 0;
                int xor_count = 0;
                int logical_ops = 0;
                int arithmetic_ops = 0;
                int loop_ops = 0;
                int index_manipulation = 0;
                
                // Analyze instructions in hot loop for memory patterns
                for (unsigned short addr = hot_start; addr < hot_end && addr < 4094; addr += 2) {
                    if (addr >= 4096) break;
                    unsigned short instr = (memory[mem_base + addr] << 8) | memory[mem_base + addr + 1];
                    unsigned char op = (instr & 0xF000) >> 12;
                    
                    if (op == 0x8) {  // Register operations
                        unsigned char subop = instr & 0x000F;
                        if (subop >= 0x1 && subop <= 0x3) {  // OR, AND, XOR
                            logical_ops++;
                            if (subop == 0x3) xor_count++;  // XOR is key for CA
                        } else if (subop == 0x4 || subop == 0x5 || subop == 0x7) {
                            arithmetic_ops++;
                        }
                    } else if (op == 0x1 || op == 0x2) {  // Jump/Call
                        loop_ops++;
                    } else if (op == 0x6 || op == 0x7) {  // Load/Add immediate
                        index_manipulation++;
                    } else if (op == 0xF) {  // Memory operations
                        unsigned char kk_val = instr & 0x00FF;
                        if (kk_val == 0x1E) {  // ADD I, Vx - critical for iteration
                            add_i_count++;
                        } else if (kk_val == 0x55) {  // LD [I], Vx - memory write
                            memory_store_count++;
                        } else if (kk_val == 0x65) {  // LD Vx, [I] - memory read
                            memory_load_count++;
                        }
                    }
                }
                
                // MEMORY-CA SCORING ALGORITHM
                
                // 1. Core memory iteration pattern (highest weight)
                if (add_i_count >= 1 && (memory_load_count >= 1 || memory_store_count >= 1)) {
                    ca_score += 40.0f;  // Strong base score for memory iteration
                    if (add_i_count >= 2) ca_score += 20.0f;  // Multiple index operations
                }
                
                // 2. Complete read-modify-write cycle (CA hallmark)
                if (memory_load_count >= 1 && memory_store_count >= 1) {
                    ca_score += 35.0f;  // Read-write cycle
                    if (logical_ops >= 1) ca_score += 25.0f;  // With computation
                    if (xor_count >= 1) ca_score += 15.0f;  // XOR is prime CA operation
                }
                
                // 3. Sequential memory access patterns
                if (sequential_memory_accesses > 10) {
                    ca_score += 20.0f;  // Sequential access pattern
                    if (sequential_memory_accesses > 50) ca_score += 15.0f;  // Heavy sequential access
                }
                
                // 4. Memory operation intensity
                if (total_memory_ops > 100) {
                    ca_score += 15.0f;  // High memory activity
                    if (total_memory_ops > 500) ca_score += 10.0f;  // Very high activity
                }
                
                // 5. Computational complexity in memory context
                if (memory_load_count >= 1 && arithmetic_ops >= 2 && logical_ops >= 1) {
                    ca_score += 20.0f;  // Complex memory-based computation
                }
                
                // 6. Loop structure with memory operations
                if (loop_ops >= 1 && total_memory_ops > 20) {
                    ca_score += 15.0f;  // Looped memory operations
                }
                
                // 7. Index manipulation diversity
                if (index_manipulation >= 2 && add_i_count >= 1) {
                    ca_score += 10.0f;  // Sophisticated indexing
                }
                
                // 8. Memory write pattern analysis
                int active_write_buckets = 0;
                for (int i = 0; i < 128; i++) {
                    if (local_memory_writes[i] > 0) active_write_buckets++;
                }
                if (active_write_buckets > 5) {
                    ca_score += 10.0f;  // Distributed memory writes
                    if (active_write_buckets > 15) ca_score += 10.0f;  // Wide memory usage
                }
                
                // 9. Execution concentration bonus
                float execution_ratio = (float)max_frequency / (float)total_frequency;
                if (execution_ratio > 0.4f) {
                    ca_score += (execution_ratio - 0.4f) * 25.0f;  // Concentrated execution
                }
                
                // Update CA detection results
                if (ca_score >= ca_threshold) {
                    ca_detected[instance] = 1;
                    ca_likelihood[instance] = fminf(ca_score, 100.0f);  // Cap at 100%
                    hot_loop_start[instance] = hot_start;
                    hot_loop_end[instance] = hot_end;
                }
            }
        }
        
        // Update timers periodically
        if (cycle % timer_update_interval == 0) {
            if (dt > 0) dt--;
            if (st > 0) st--;
        }
    }
    
    // Write back local state
    program_counters[instance] = pc;
    index_registers[instance] = index_reg;
    stack_pointers[instance] = sp;
    delay_timers[instance] = dt;
    sound_timers[instance] = st;
    
    // Update statistics
    instructions_executed[instance] += local_instructions;
    memory_operations[instance] += local_memory_operations;
    
    // Copy local frequency data to global memory
    for (int i = 0; i < 256; i++) {
        pc_frequency[pc_freq_base + i] = local_pc_frequency[i];
    }
    for (int i = 0; i < 128; i++) {
        memory_write_frequency[mem_write_base + i] = local_memory_writes[i];
    }
}
'''

class MemoryCADetector:
    """
    GPU-optimized CHIP-8 emulator focused exclusively on memory-based CA detection
    Optimized for GTX 1070 and similar GPUs
    """
    
    def __init__(self, num_instances: int, quirks: dict = None, 
                 ca_detection_interval: int = 500, ca_threshold: float = 30.0):
        
        # Get current GPU info
        current_device = cp.cuda.Device()
        device_id = current_device.id
        props = cp.cuda.runtime.getDeviceProperties(device_id)
        gpu_name = props['name'].decode()
        gpu_memory_gb = props['totalGlobalMem'] / 1024**3
        
        self.num_instances = num_instances
        
        # CHIP-8 Quirks configuration
        self.quirks = quirks or {
            'memory': True,      
            'jumping': True,     
            'logic': True,       
        }
        
        # CA detection parameters
        self.ca_detection_interval = ca_detection_interval
        self.ca_threshold = ca_threshold
        
        # Compile the memory-CA kernel
        self.memory_ca_kernel = cp.RawKernel(MEMORY_CA_KERNEL_SOURCE, 'chip8_memory_ca_kernel')
        
        # GPU-optimized block/grid sizes
        if gpu_memory_gb >= 10:  # High-end GPU
            self.block_size = min(512, num_instances)
        elif gpu_memory_gb >= 6:  # Mid-range GPU (GTX 1070 level)
            self.block_size = min(256, num_instances)
        else:  # Conservative for smaller GPUs
            self.block_size = min(128, num_instances)
            
        self.grid_size = (num_instances + self.block_size - 1) // self.block_size
        
        print(f"🚀 Memory-CA Detector: {num_instances:,} instances on {gpu_name}")
        print(f"   GPU memory: {gpu_memory_gb:.1f} GB")
        print(f"   Block size: {self.block_size}, Grid size: {self.grid_size}")
        print(f"   Memory-CA detection interval: {ca_detection_interval} cycles, threshold: {ca_threshold}%")
        print(f"   Focus: Memory patterns only, display operations ignored")
        
        # Initialize all state
        self._initialize_state()
        self._initialize_stats()
        self._initialize_memory_ca_detection()
    
    def _initialize_state(self):
        """Initialize all state arrays"""
        # Memory: (instances, memory_size)
        self.memory = cp.zeros((self.num_instances, MEMORY_SIZE), dtype=cp.uint8)
        
        # Registers: (instances, 16)
        self.registers = cp.zeros((self.num_instances, REGISTER_COUNT), dtype=cp.uint8)
        
        # System state
        self.index_register = cp.zeros(self.num_instances, dtype=cp.uint16)
        self.program_counter = cp.full(self.num_instances, PROGRAM_START, dtype=cp.uint16)
        self.stack_pointer = cp.zeros(self.num_instances, dtype=cp.uint8)
        self.stack = cp.zeros((self.num_instances, STACK_SIZE), dtype=cp.uint16)
        
        # Timers
        self.delay_timer = cp.zeros(self.num_instances, dtype=cp.uint8)
        self.sound_timer = cp.zeros(self.num_instances, dtype=cp.uint8)
        
        # State flags
        self.crashed = cp.zeros(self.num_instances, dtype=cp.uint8)
        self.halted = cp.zeros(self.num_instances, dtype=cp.uint8)
        
        # Random number state for RND instruction
        self.rng_state = cp.random.randint(1, 2**32, size=self.num_instances, dtype=cp.uint32)
        
        # Load font into all instances
        font_data = cp.tile(CHIP8_FONT, (self.num_instances, 1))
        self.memory[:, FONT_START:FONT_START + len(CHIP8_FONT)] = font_data
    
    def _initialize_stats(self):
        """Initialize minimal statistics arrays for memory-CA focus"""
        self.stats = {
            'instructions_executed': cp.zeros(self.num_instances, dtype=cp.uint32),
            'memory_operations': cp.zeros(self.num_instances, dtype=cp.uint32),
        }
    
    def _initialize_memory_ca_detection(self):
        """Initialize memory-CA detection arrays"""
        # CA detection results
        self.ca_detected = cp.zeros(self.num_instances, dtype=cp.uint8)
        self.ca_likelihood = cp.zeros(self.num_instances, dtype=cp.float32)
        self.hot_loop_start = cp.zeros(self.num_instances, dtype=cp.uint16)
        self.hot_loop_end = cp.zeros(self.num_instances, dtype=cp.uint16)
        
        # Memory tracking (instances x buckets)
        self.pc_frequency = cp.zeros((self.num_instances, 256), dtype=cp.uint32)
        self.memory_write_frequency = cp.zeros((self.num_instances, 128), dtype=cp.uint32)
    
    def reset(self):
        """Reset all instances"""
        self.memory.fill(0)
        self.registers.fill(0)
        self.index_register.fill(0)
        self.program_counter.fill(PROGRAM_START)
        self.stack_pointer.fill(0)
        self.stack.fill(0)
        self.delay_timer.fill(0)
        self.sound_timer.fill(0)
        
        # Reset memory-CA detection
        self.ca_detected.fill(0)
        self.ca_likelihood.fill(0.0)
        self.hot_loop_start.fill(0)
        self.hot_loop_end.fill(0)
        self.pc_frequency.fill(0)
        self.memory_write_frequency.fill(0)
        
        # Reload font
        font_data = cp.tile(CHIP8_FONT, (self.num_instances, 1))
        self.memory[:, FONT_START:FONT_START + len(CHIP8_FONT)] = font_data
        
        # Reset stats
        for stat_array in self.stats.values():
            stat_array.fill(0)
        
        # Reset RNG state
        self.rng_state = cp.random.randint(1, 2**32, size=self.num_instances, dtype=cp.uint32)
    
    def load_roms(self, rom_data_list: List[Union[bytes, np.ndarray]]):
        """Load ROMs into instances"""
        if not rom_data_list:
            raise ValueError("No ROM data provided")
        
        for i in range(self.num_instances):
            rom_data = rom_data_list[i % len(rom_data_list)]
            
            if isinstance(rom_data, np.ndarray):
                rom_bytes = rom_data
            else:
                rom_bytes = np.frombuffer(rom_data, dtype=np.uint8)
            
            if len(rom_bytes) > MEMORY_SIZE - PROGRAM_START:
                raise ValueError(f"ROM {i} too large: {len(rom_bytes)} bytes")
            
            rom_end = PROGRAM_START + len(rom_bytes)
            self.memory[i, PROGRAM_START:rom_end] = cp.array(rom_bytes)
        
        print(f"Loaded ROMs into {self.num_instances:,} instances for memory-CA detection")
    
    def load_single_rom(self, rom_data: Union[bytes, np.ndarray]):
        """Load the same ROM into all instances"""
        if isinstance(rom_data, np.ndarray):
            rom_bytes = rom_data
        else:
            rom_bytes = np.frombuffer(rom_data, dtype=np.uint8)
        
        if len(rom_bytes) > MEMORY_SIZE - PROGRAM_START:
            raise ValueError(f"ROM too large: {len(rom_bytes)} bytes")
        
        # Broadcast ROM to all instances
        rom_gpu = cp.array(rom_bytes)
        rom_end = PROGRAM_START + len(rom_bytes)
        self.memory[:, PROGRAM_START:rom_end] = rom_gpu[None, :]
        
        print(f"Loaded single ROM into {self.num_instances:,} instances for memory-CA detection")
    
    def run(self, cycles: int = 5000, timer_update_interval: int = 16):
        """Run the memory-CA detection kernel"""
        print(f"🚀 Launching memory-CA detection kernel for {cycles:,} cycles...")
        
        # Reset CA detection for this run
        self.ca_detected.fill(0)
        self.ca_likelihood.fill(0.0)
        self.hot_loop_start.fill(0)
        self.hot_loop_end.fill(0)
        self.pc_frequency.fill(0)
        self.memory_write_frequency.fill(0)
        
        start_time = time.time()
        
        # Launch the memory-CA detection kernel
        self.memory_ca_kernel(
            (self.grid_size,), (self.block_size,),
            (
                # State arrays
                self.memory,
                self.registers,
                self.index_register,
                self.program_counter,
                self.stack_pointer,
                self.stack,
                self.delay_timer,
                self.sound_timer,
                
                # State flags
                self.crashed,
                self.halted,
                
                # Minimal statistics
                self.stats['instructions_executed'],
                self.stats['memory_operations'],
                
                # RNG state
                self.rng_state,
                
                # Memory-CA detection arrays
                self.ca_detected,
                self.ca_likelihood,
                self.hot_loop_start,
                self.hot_loop_end,
                self.pc_frequency,
                self.memory_write_frequency,
                
                # Parameters
                self.num_instances,
                cycles,
                timer_update_interval,
                
                # Memory-CA detection parameters
                self.ca_detection_interval,
                self.ca_threshold,
                
                # Quirks
                1 if self.quirks['memory'] else 0,
                1 if self.quirks['jumping'] else 0,
                1 if self.quirks['logic'] else 0
            )
        )
        
        # Synchronize GPU
        cp.cuda.Stream.null.synchronize()
        
        end_time = time.time()
        execution_time = end_time - start_time
        
        total_instructions = int(cp.sum(self.stats['instructions_executed']))
        total_memory_ops = int(cp.sum(self.stats['memory_operations']))
        instructions_per_second = total_instructions / execution_time if execution_time > 0 else 0
        
        print(f"🚀 Memory-CA kernel execution: {execution_time:.4f}s")
        print(f"Total instructions: {total_instructions:,}")
        print(f"Total memory operations: {total_memory_ops:,}")
        print(f"Instructions/second: {instructions_per_second:,.0f}")
        print(f"Memory operations/second: {total_memory_ops / execution_time:,.0f}")
        
        # Report memory-CA detection results
        ca_count = int(cp.sum(self.ca_detected))
        if ca_count > 0:
            max_likelihood = float(cp.max(self.ca_likelihood))
            print(f"🔬 MEMORY-CA DETECTION: Found {ca_count} potential memory-CA patterns!")
            print(f"   Max memory-CA likelihood: {max_likelihood:.1f}%")
            
            # Report top CA candidates
            ca_results = self.get_ca_results()
            if ca_results['ca_instances']:
                top_5 = ca_results['ca_instances'][:5]
                print("   Top memory-CA candidates:")
                for i, ca in enumerate(top_5, 1):
                    print(f"   {i}. Instance {ca['instance_id']:04d}: {ca['ca_likelihood']:.1f}% likelihood")
        else:
            print("   No memory-CA patterns detected in this batch")
    
    def get_ca_results(self) -> Dict:
        """Get memory-CA detection results"""
        ca_detected_np = cp.asnumpy(self.ca_detected).astype(bool)
        ca_likelihood_np = cp.asnumpy(self.ca_likelihood)
        hot_start_np = cp.asnumpy(self.hot_loop_start)
        hot_end_np = cp.asnumpy(self.hot_loop_end)
        
        ca_instances = []
        for i in range(self.num_instances):
            if ca_detected_np[i]:
                ca_instances.append({
                    'instance_id': i,
                    'ca_likelihood': float(ca_likelihood_np[i]),
                    'hot_loop_range': (int(hot_start_np[i]), int(hot_end_np[i])),
                    'memory_operations': int(self.stats['memory_operations'][i])
                })
        
        # Sort by likelihood (highest first)
        ca_instances.sort(key=lambda x: x['ca_likelihood'], reverse=True)
        
        return {
            'ca_detected_count': len(ca_instances),
            'ca_instances': ca_instances,
            'max_ca_likelihood': float(np.max(ca_likelihood_np)) if len(ca_instances) > 0 else 0.0
        }
    
    def get_memory_analysis(self, instance_id: int) -> Dict:
        """Get detailed memory analysis for a specific instance"""
        if instance_id >= self.num_instances:
            raise ValueError(f"Instance {instance_id} out of range")
        
        # Get memory write patterns
        memory_writes = cp.asnumpy(self.memory_write_frequency[instance_id])
        pc_freq = cp.asnumpy(self.pc_frequency[instance_id])
        
        # Analyze memory usage
        active_memory_regions = np.where(memory_writes > 0)[0]
        total_memory_writes = np.sum(memory_writes)
        
        # Find most active memory regions
        top_regions = np.argsort(memory_writes)[-10:][::-1]
        
        return {
            'instance_id': instance_id,
            'ca_detected': bool(self.ca_detected[instance_id]),
            'ca_likelihood': float(self.ca_likelihood[instance_id]),
            'total_memory_writes': int(total_memory_writes),
            'active_memory_regions': len(active_memory_regions),
            'top_memory_regions': [(int(r), int(memory_writes[r])) for r in top_regions if memory_writes[r] > 0],
            'hot_loop_range': (int(self.hot_loop_start[instance_id]), int(self.hot_loop_end[instance_id])),
            'instructions_executed': int(self.stats['instructions_executed'][instance_id]),
            'memory_operations': int(self.stats['memory_operations'][instance_id])
        }
    
    def get_aggregate_stats(self) -> Dict[str, Union[int, float]]:
        """Get aggregate statistics including memory-CA results"""
        aggregate = {}
        
        for key, arr in self.stats.items():
            aggregate[f"total_{key}"] = int(cp.sum(arr))
            aggregate[f"mean_{key}"] = float(cp.mean(arr))
            aggregate[f"max_{key}"] = int(cp.max(arr))
            aggregate[f"min_{key}"] = int(cp.min(arr))
        
        aggregate['active_instances'] = int(cp.sum(~self.crashed))
        aggregate['crashed_instances'] = int(cp.sum(self.crashed))
        aggregate['total_instances'] = self.num_instances
        
        # Add memory-CA detection stats
        aggregate['ca_detected_count'] = int(cp.sum(self.ca_detected))
        aggregate['max_ca_likelihood'] = float(cp.max(self.ca_likelihood))
        
        # Memory-specific stats
        total_memory_writes = int(cp.sum(self.memory_write_frequency))
        aggregate['total_memory_writes'] = total_memory_writes
        aggregate['mean_memory_writes_per_instance'] = total_memory_writes / self.num_instances
        
        return aggregate
    
    def print_aggregate_stats(self):
        """Print aggregate statistics including memory-CA detection results"""
        stats = self.get_aggregate_stats()
        
        print("🚀 Memory-CA Detector Statistics:")
        print("=" * 50)
        print(f"Total instances: {stats['total_instances']}")
        print(f"Active instances: {stats['active_instances']}")
        print(f"Crashed instances: {stats['crashed_instances']}")
        print()
        
        print("Execution totals:")
        print(f"Instructions executed: {stats['total_instructions_executed']:,}")
        print(f"Memory operations: {stats['total_memory_operations']:,}")
        print(f"Memory writes tracked: {stats['total_memory_writes']:,}")
        print()
        
        print("Per-instance averages:")
        print(f"Instructions: {stats['mean_instructions_executed']:.1f}")
        print(f"Memory operations: {stats['mean_memory_operations']:.1f}")
        print(f"Memory writes: {stats['mean_memory_writes_per_instance']:.1f}")
        print()
        
        print("Memory-CA Detection Results:")
        print(f"Memory-CA patterns detected: {stats['ca_detected_count']}")
        print(f"Max memory-CA likelihood: {stats['max_ca_likelihood']:.1f}%")
        
        if stats['ca_detected_count'] > 0:
            memory_op_ratio = stats['total_memory_operations'] / stats['total_instructions_executed'] * 100
            print(f"Memory operation ratio: {memory_op_ratio:.2f}%")
    
    def export_ca_roms(self, output_dir: str, max_exports: int = 10):
        """Export ROM data for detected CA patterns"""
        import os
        
        os.makedirs(output_dir, exist_ok=True)
        
        ca_results = self.get_ca_results()
        if not ca_results['ca_instances']:
            print("No CA patterns to export")
            return []
        
        exported_files = []
        instances_to_export = ca_results['ca_instances'][:max_exports]
        
        for ca_info in instances_to_export:
            instance_id = ca_info['instance_id']
            likelihood = ca_info['ca_likelihood']
            
            # Extract ROM data (everything from PROGRAM_START onwards)
            rom_data = cp.asnumpy(self.memory[instance_id, PROGRAM_START:])
            
            # Find actual end of ROM (first long stretch of zeros)
            rom_end = len(rom_data)
            zero_count = 0
            for i in range(len(rom_data)):
                if rom_data[i] == 0:
                    zero_count += 1
                    if zero_count > 32:  # 32 consecutive zeros = end of ROM
                        rom_end = i - 31
                        break
                else:
                    zero_count = 0
            
            rom_data = rom_data[:rom_end]
            
            filename = f"memory_ca_instance_{instance_id:04d}_likelihood_{likelihood:.1f}.ch8"
            filepath = os.path.join(output_dir, filename)
            
            with open(filepath, 'wb') as f:
                f.write(rom_data.tobytes())
            
            exported_files.append(filepath)
            print(f"Exported memory-CA ROM: {filename} ({len(rom_data)} bytes, {likelihood:.1f}% likelihood)")
        
        return exported_files


# Backward compatibility and convenience aliases
ParallelChip8Emulator = MemoryCADetector
MegaKernelChip8Emulator = MemoryCADetector