"""
Mega-Kernel CHIP-8 Emulator
Everything runs in a single CUDA kernel for maximum performance
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

# The mega-kernel that does everything
MEGA_KERNEL_SOURCE = r'''
extern "C" __global__
void chip8_mega_kernel(
    // State arrays
    unsigned char* memory,              // [instances][4096]
    unsigned char* displays,            // [instances][32][64] 
    unsigned char* registers,           // [instances][16]
    unsigned short* index_registers,    // [instances]
    unsigned short* program_counters,   // [instances]
    unsigned char* stack_pointers,      // [instances]
    unsigned short* stacks,             // [instances][16]
    unsigned char* delay_timers,        // [instances]
    unsigned char* sound_timers,        // [instances]
    unsigned char* keypad,              // [instances][16]
    
    // State flags
    unsigned char* crashed,             // [instances]
    unsigned char* halted,              // [instances]
    unsigned char* waiting_for_key,     // [instances]
    unsigned char* key_registers,       // [instances]
    
    // Statistics arrays
    unsigned int* instructions_executed,    // [instances]
    unsigned int* display_writes,           // [instances]
    unsigned int* pixels_drawn,             // [instances]
    unsigned int* pixels_erased,            // [instances]
    unsigned int* sprite_collisions,        // [instances]
    
    // Random number state
    unsigned int* rng_state,            // [instances] - for RND instruction
    
    // Execution parameters
    int num_instances,
    int cycles_to_run,
    int timer_update_interval,
    
    // Quirks
    int quirk_memory,
    int quirk_jumping,
    int quirk_logic
) {
    int instance = blockIdx.x * blockDim.x + threadIdx.x;
    if (instance >= num_instances) return;
    
    // Calculate base indices for this instance
    int mem_base = instance * 4096;
    int display_base = instance * 32 * 64;
    int reg_base = instance * 16;
    int stack_base = instance * 16;
    int keypad_base = instance * 16;
    
    // Local state (registers for better performance)
    unsigned short pc = program_counters[instance];
    unsigned short index_reg = index_registers[instance];
    unsigned char sp = stack_pointers[instance];
    unsigned char dt = delay_timers[instance];
    unsigned char st = sound_timers[instance];
    
    // Statistics
    unsigned int local_instructions = 0;
    unsigned int local_display_writes = 0;
    unsigned int local_pixels_drawn = 0;
    unsigned int local_pixels_erased = 0;
    unsigned int local_collisions = 0;
    
    // Check if this instance is active
    if (crashed[instance] || halted[instance]) {
        return; // Skip crashed/halted instances
    }
    
    // Main execution loop
    for (int cycle = 0; cycle < cycles_to_run; cycle++) {
        // Skip if waiting for key
        if (waiting_for_key[instance]) {
            continue;
        }
        
        // Check PC bounds
        if (pc >= 4096 - 1) {
            crashed[instance] = 1;
            break;
        }
        
        // Fetch instruction
        unsigned char high_byte = memory[mem_base + pc];
        unsigned char low_byte = memory[mem_base + pc + 1];
        unsigned short instruction = (high_byte << 8) | low_byte;
        
        // Increment PC
        pc += 2;
        local_instructions++;
        
        // Decode instruction
        unsigned char opcode = (instruction & 0xF000) >> 12;
        unsigned char x = (instruction & 0x0F00) >> 8;
        unsigned char y = (instruction & 0x00F0) >> 4;
        unsigned char n = instruction & 0x000F;
        unsigned char kk = instruction & 0x00FF;
        unsigned short nnn = instruction & 0x0FFF;
        
        // Execute instruction
        switch (opcode) {
            case 0x0:
                if (instruction == 0x00E0) {
                    // CLS - Clear display
                    for (int i = 0; i < 32 * 64; i++) {
                        displays[display_base + i] = 0;
                    }
                    local_display_writes++;
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
                // SE Vx, byte - Skip if equal
                if (registers[reg_base + x] == kk) {
                    pc += 2;
                }
                break;
                
            case 0x4:
                // SNE Vx, byte - Skip if not equal
                if (registers[reg_base + x] != kk) {
                    pc += 2;
                }
                break;
                
            case 0x5:
                // SE Vx, Vy - Skip if registers equal
                if (n == 0 && registers[reg_base + x] == registers[reg_base + y]) {
                    pc += 2;
                }
                break;
                
            case 0x6:
                // LD Vx, byte - Load byte into register
                registers[reg_base + x] = kk;
                break;
                
            case 0x7:
                // ADD Vx, byte - Add byte to register
                registers[reg_base + x] = (registers[reg_base + x] + kk) & 0xFF;
                break;
                
            case 0x8:
                // Register operations
                {
                    unsigned char vx = registers[reg_base + x];
                    unsigned char vy = registers[reg_base + y];
                    unsigned char result = 0;
                    unsigned char flag = 0;
                    
                    switch (n) {
                        case 0x0: // LD Vx, Vy
                            result = vy;
                            break;
                        case 0x1: // OR Vx, Vy
                            result = vx | vy;
                            if (quirk_logic) flag = 0;
                            break;
                        case 0x2: // AND Vx, Vy
                            result = vx & vy;
                            if (quirk_logic) flag = 0;
                            break;
                        case 0x3: // XOR Vx, Vy
                            result = vx ^ vy;
                            if (quirk_logic) flag = 0;
                            break;
                        case 0x4: // ADD Vx, Vy
                            {
                                int sum = vx + vy;
                                result = sum & 0xFF;
                                flag = (sum > 255) ? 1 : 0;
                            }
                            break;
                        case 0x5: // SUB Vx, Vy
                            result = (vx - vy) & 0xFF;
                            flag = (vx >= vy) ? 1 : 0;
                            break;
                        case 0x6: // SHR Vx
                            result = vx >> 1;
                            flag = vx & 0x1;
                            break;
                        case 0x7: // SUBN Vx, Vy
                            result = (vy - vx) & 0xFF;
                            flag = (vy >= vx) ? 1 : 0;
                            break;
                        case 0xE: // SHL Vx
                            result = (vx << 1) & 0xFF;
                            flag = (vx & 0x80) ? 1 : 0;
                            break;
                        default:
                            crashed[instance] = 1;
                            continue;
                    }
                    
                    registers[reg_base + x] = result;
                    if (n == 0x1 || n == 0x2 || n == 0x3 || n == 0x4 || n == 0x5 || n == 0x6 || n == 0x7 || n == 0xE) {
                        registers[reg_base + 0xF] = flag;
                    }
                }
                break;
                
            case 0x9:
                // SNE Vx, Vy - Skip if registers not equal
                if (n == 0 && registers[reg_base + x] != registers[reg_base + y]) {
                    pc += 2;
                }
                break;
                
            case 0xA:
                // LD I, addr - Load address into I
                index_reg = nnn;
                break;
                
            case 0xB:
                // JP V0, addr - Jump to address plus V0
                if (quirk_jumping) {
                    pc = nnn + registers[reg_base + ((nnn & 0xF00) >> 8)];
                } else {
                    pc = nnn + registers[reg_base + 0];
                }
                break;
                
            case 0xC:
                // RND Vx, byte - Random number AND byte
                {
                    // Simple LCG random number generator
                    rng_state[instance] = rng_state[instance] * 1664525 + 1013904223;
                    unsigned char random_byte = (rng_state[instance] >> 16) & 0xFF;
                    registers[reg_base + x] = random_byte & kk;
                }
                break;
                
            case 0xD:
                // DRW Vx, Vy, nibble - Draw sprite
                {
                    unsigned char vx = registers[reg_base + x] % 64;
                    unsigned char vy = registers[reg_base + y] % 32;
                    registers[reg_base + 0xF] = 0; // Clear collision flag
                    
                    for (int row = 0; row < n; row++) {
                        if (vy + row >= 32) break;
                        if (index_reg + row >= 4096) break;
                        
                        unsigned char sprite_byte = memory[mem_base + index_reg + row];
                        
                        for (int col = 0; col < 8; col++) {
                            if (vx + col >= 64) break;
                            
                            if (sprite_byte & (0x80 >> col)) {
                                int pixel_idx = display_base + (vy + row) * 64 + (vx + col);
                                
                                if (displays[pixel_idx]) {
                                    registers[reg_base + 0xF] = 1; // Collision
                                    local_collisions++;
                                    local_pixels_erased++;
                                } else {
                                    local_pixels_drawn++;
                                }
                                
                                displays[pixel_idx] ^= 1;
                            }
                        }
                    }
                    local_display_writes++;
                }
                break;
                
            case 0xE:
                // Key operations
                {
                    unsigned char key = registers[reg_base + x] & 0xF;
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
                // Timer and misc operations
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
                    case 0x29: // LD F, Vx
                        {
                            unsigned char digit = registers[reg_base + x] & 0xF;
                            index_reg = 0x50 + digit * 5; // Font location
                        }
                        break;
                    case 0x33: // LD B, Vx - BCD
                        {
                            unsigned char value = registers[reg_base + x];
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
                            }
                        }
                        if (quirk_memory) {
                            index_reg = (index_reg + x + 1) & 0xFFFF;
                        }
                        break;
                    case 0x65: // LD Vx, [I] - Load registers
                        for (int i = 0; i <= x; i++) {
                            if (index_reg + i < 4096) {
                                registers[reg_base + i] = memory[mem_base + index_reg + i];
                            }
                        }
                        if (quirk_memory) {
                            index_reg = (index_reg + x + 1) & 0xFFFF;
                        }
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
    display_writes[instance] += local_display_writes;
    pixels_drawn[instance] += local_pixels_drawn;
    pixels_erased[instance] += local_pixels_erased;
    sprite_collisions[instance] += local_collisions;
}
'''

class MegaKernelChip8Emulator:
    """
    Ultimate performance CHIP-8 emulator with everything in a single CUDA kernel
    """
    
    def __init__(self, num_instances: int, quirks: dict = None):
        self.num_instances = num_instances
        
        # CHIP-8 Quirks configuration
        self.quirks = quirks or {
            'memory': True,      
            'display_wait': False, 
            'jumping': True,     
            'shifting': False,   
            'logic': True,       
        }
        
        # Compile the mega kernel
        self.mega_kernel = cp.RawKernel(MEGA_KERNEL_SOURCE, 'chip8_mega_kernel')
        
        # Calculate optimal block/grid sizes
        # This is optimized for the GTX 1070;
        # Tested 128, 256, 512 block sizes, 256 had highest throughput
        self.block_size = min(256, num_instances)
        self.grid_size = (num_instances + self.block_size - 1) // self.block_size
        
        print(f"Mega-Kernel CHIP-8: {num_instances} instances, block_size={self.block_size}, grid_size={self.grid_size}")
        
        # Initialize all state
        self._initialize_state()
        self._initialize_stats()
    
    def _initialize_state(self):
        """Initialize all state arrays"""
        # Memory: (instances, memory_size)
        self.memory = cp.zeros((self.num_instances, MEMORY_SIZE), dtype=cp.uint8)
        
        # Display: (instances, height, width) - flattened for kernel
        self.display = cp.zeros((self.num_instances, DISPLAY_HEIGHT * DISPLAY_WIDTH), dtype=cp.uint8)
        
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
        
        # Input
        self.keypad = cp.zeros((self.num_instances, KEYPAD_SIZE), dtype=cp.uint8)
        
        # State flags
        self.crashed = cp.zeros(self.num_instances, dtype=cp.uint8)
        self.halted = cp.zeros(self.num_instances, dtype=cp.uint8)
        self.waiting_for_key = cp.zeros(self.num_instances, dtype=cp.uint8)
        self.key_register = cp.zeros(self.num_instances, dtype=cp.uint8)
        
        # Random number state for RND instruction
        self.rng_state = cp.random.randint(1, 2**32, size=self.num_instances, dtype=cp.uint32)
        
        # Load font into all instances
        font_data = cp.tile(CHIP8_FONT, (self.num_instances, 1))
        self.memory[:, FONT_START:FONT_START + len(CHIP8_FONT)] = font_data
    
    def _initialize_stats(self):
        """Initialize statistics arrays"""
        self.stats = {
            'instructions_executed': cp.zeros(self.num_instances, dtype=cp.uint32),
            'display_writes': cp.zeros(self.num_instances, dtype=cp.uint32),
            'pixels_drawn': cp.zeros(self.num_instances, dtype=cp.uint32),
            'pixels_erased': cp.zeros(self.num_instances, dtype=cp.uint32),
            'sprite_collisions': cp.zeros(self.num_instances, dtype=cp.uint32),
        }
    
    def reset(self):
        """Reset all instances"""
        self.memory.fill(0)
        self.display.fill(0)
        self.registers.fill(0)
        self.index_register.fill(0)
        self.program_counter.fill(PROGRAM_START)
        self.stack_pointer.fill(0)
        self.stack.fill(0)
        self.delay_timer.fill(0)
        self.sound_timer.fill(0)
        self.keypad.fill(0)
        self.crashed.fill(0)
        self.halted.fill(0)
        self.waiting_for_key.fill(0)
        self.key_register.fill(0)
        
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
        
        print(f"Loaded ROMs into {self.num_instances} instances")
    
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
        
        print(f"Loaded single ROM into {self.num_instances} instances")
    
    def run(self, cycles: int = 1000, timer_update_interval: int = 16):
        """Run the mega kernel for specified cycles"""
        print(f"Launching mega-kernel for {cycles} cycles...")
        
        start_time = time.time()
        
        # Launch the mega kernel
        self.mega_kernel(
            (self.grid_size,), (self.block_size,),
            (
                # State arrays
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
                
                # Statistics
                self.stats['instructions_executed'],
                self.stats['display_writes'],
                self.stats['pixels_drawn'],
                self.stats['pixels_erased'],
                self.stats['sprite_collisions'],
                
                # RNG state
                self.rng_state,
                
                # Parameters
                self.num_instances,
                cycles,
                timer_update_interval,
                
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
        instructions_per_second = total_instructions / execution_time if execution_time > 0 else 0
        
        print(f"Mega-kernel execution: {execution_time:.4f}s")
        print(f"Total instructions: {total_instructions:,}")
        print(f"Instructions/second: {instructions_per_second:,.0f}")
    
    def get_displays(self, instance_ids: Optional[List[int]] = None) -> cp.ndarray:
        """Get display data reshaped back to 2D"""
        if instance_ids is None:
            displays = self.display.copy()
        else:
            displays = self.display[instance_ids].copy()
        
        # Reshape back to (instances, height, width)
        return displays.reshape(-1, DISPLAY_HEIGHT, DISPLAY_WIDTH)
    
    def get_displays_as_images(self, instance_ids: Optional[List[int]] = None, scale: int = 8) -> np.ndarray:
        """Get display data as scaled images"""
        displays = self.get_displays(instance_ids)
        displays_np = cp.asnumpy(displays)
        
        if len(displays_np.shape) == 2:
            displays_np = displays_np[None, ...]
        
        scaled_displays = []
        for display in displays_np:
            scaled = np.repeat(np.repeat(display, scale, axis=0), scale, axis=1)
            scaled_displays.append(scaled * 255)
        
        return np.array(scaled_displays, dtype=np.uint8)
    
    def get_aggregate_stats(self) -> Dict[str, Union[int, float]]:
        """Get aggregate statistics"""
        aggregate = {}
        
        for key, arr in self.stats.items():
            aggregate[f"total_{key}"] = int(cp.sum(arr))
            aggregate[f"mean_{key}"] = float(cp.mean(arr))
            aggregate[f"max_{key}"] = int(cp.max(arr))
            aggregate[f"min_{key}"] = int(cp.min(arr))
        
        aggregate['active_instances'] = int(cp.sum(~self.crashed))
        aggregate['crashed_instances'] = int(cp.sum(self.crashed))
        aggregate['waiting_instances'] = int(cp.sum(self.waiting_for_key))
        aggregate['total_instances'] = self.num_instances
        
        return aggregate
    
    def print_aggregate_stats(self):
        """Print aggregate statistics"""
        stats = self.get_aggregate_stats()
        
        print("Mega-Kernel CHIP-8 Emulator Statistics:")
        print("=" * 50)
        print(f"Total instances: {stats['total_instances']}")
        print(f"Active instances: {stats['active_instances']}")
        print(f"Crashed instances: {stats['crashed_instances']}")
        print(f"Waiting for key: {stats['waiting_instances']}")
        print()
        
        print("Execution totals:")
        print(f"Instructions executed: {stats['total_instructions_executed']:,}")
        print(f"Display writes: {stats['total_display_writes']:,}")
        print(f"Pixels drawn: {stats['total_pixels_drawn']:,}")
        print(f"Sprite collisions: {stats['total_sprite_collisions']:,}")
        print()
        
        print("Per-instance averages:")
        print(f"Instructions: {stats['mean_instructions_executed']:.1f}")
        print(f"Display writes: {stats['mean_display_writes']:.1f}")
        print(f"Pixels drawn: {stats['mean_pixels_drawn']:.1f}")
        print(f"Collisions: {stats['mean_sprite_collisions']:.1f}")
    
    def save_displays_as_pngs(self, output_dir: str, instance_ids: Optional[List[int]] = None, 
                             scale: int = 8, prefix: str = "display"):
        """Save display outputs as PNG files"""
        import os
        from PIL import Image
        
        os.makedirs(output_dir, exist_ok=True)
        
        images = self.get_displays_as_images(instance_ids, scale)
        
        if instance_ids is None:
            instance_ids = list(range(self.num_instances))
        elif len(images) != len(instance_ids):
            images = images[instance_ids]
        
        saved_files = []
        for i, (instance_id, image_data) in enumerate(zip(instance_ids, images)):
            filename = f"{prefix}_instance_{instance_id:04d}.png"
            filepath = os.path.join(output_dir, filename)
            
            # Convert grayscale to RGB for PNG
            if len(image_data.shape) == 2:
                rgb_image = np.stack([image_data] * 3, axis=-1)
            else:
                rgb_image = image_data
            
            img = Image.fromarray(rgb_image.astype(np.uint8))
            img.save(filepath)
            saved_files.append(filepath)
        
        print(f"Saved {len(saved_files)} display images to {output_dir}")
        return saved_files
    
    def set_keys(self, instance_keys: Dict[int, Dict[int, bool]]):
        """Set key states for specific instances"""
        for instance_id, keys in instance_keys.items():
            if 0 <= instance_id < self.num_instances:
                for key_id, pressed in keys.items():
                    if 0 <= key_id <= 0xF:
                        self.keypad[instance_id, key_id] = 1 if pressed else 0
                        
                        # Handle key waiting
                        if self.waiting_for_key[instance_id] and pressed:
                            self.registers[instance_id, int(self.key_register[instance_id])] = key_id
                            self.waiting_for_key[instance_id] = False


# Backward compatibility
ParallelChip8Emulator = MegaKernelChip8Emulator