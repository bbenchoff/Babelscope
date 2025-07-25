"""
CUDA-Optimized Sorting Algorithm Detector for Babelscope - PERFORMANCE OPTIMIZED
Optimized for maximum GPU utilization and throughput
"""

import cupy as cp
import numpy as np
import time
import os
from typing import List, Dict, Tuple, Optional, Union
import argparse

# Sorting detection constants
SORT_ARRAY_START = 0x300
SORT_ARRAY_SIZE = 8
SORT_CHECK_INTERVAL = 500

# Optimized CUDA kernel with better memory access patterns
CUDA_SORTING_KERNEL_OPTIMIZED = r'''
extern "C" __global__
void chip8_sorting_kernel_optimized(
    // CHIP-8 state arrays - optimized layout
    unsigned char* memory,                  // [instances][4096] - coalesced access
    unsigned char* registers,               // [instances][16]
    unsigned short* index_registers,        // [instances]
    unsigned short* program_counters,       // [instances]
    unsigned char* stack_pointers,          // [instances]
    unsigned short* stacks,                 // [instances][16]
    unsigned char* delay_timers,            // [instances]
    unsigned char* sound_timers,            // [instances]
    unsigned char* crashed,                 // [instances]
    unsigned char* halted,                  // [instances]
    unsigned int* rng_state,                // [instances]
    
    // Sorting detection results - packed for better memory efficiency
    unsigned char* sort_detected,           // [instances]
    unsigned char* sort_direction,          // [instances]
    unsigned int* sort_cycle,               // [instances]
    unsigned char* array_accessed,          // [instances]
    unsigned int* array_reads,              // [instances]
    unsigned int* array_writes,             // [instances]
    unsigned int* comparison_operations,    // [instances]
    unsigned int* swap_operations,          // [instances]
    
    // Statistics
    unsigned int* instructions_executed,    // [instances]
    unsigned int* memory_operations,        // [instances]
    
    // Parameters
    int num_instances,
    int cycles_to_run,
    int timer_update_interval,
    int sort_check_interval,
    
    // Quirks
    int quirk_memory,
    int quirk_jumping,
    int quirk_logic
) {
    // Optimized thread indexing
    int instance = blockIdx.x * blockDim.x + threadIdx.x;
    int warp_id = threadIdx.x / 32;
    int lane_id = threadIdx.x % 32;
    
    if (instance >= num_instances) return;
    
    // Pre-calculate memory offsets for better cache usage
    int mem_base = instance * 4096;
    int reg_base = instance * 16;
    int stack_base = instance * 16;
    
    // Use shared memory for frequently accessed data
    __shared__ unsigned char shared_font[80];
    if (threadIdx.x < 80) {
        // Load font data into shared memory
        unsigned char font_data[80] = {
            0xF0, 0x90, 0x90, 0x90, 0xF0, 0x20, 0x60, 0x20, 0x20, 0x70,
            0xF0, 0x10, 0xF0, 0x80, 0xF0, 0xF0, 0x10, 0xF0, 0x10, 0xF0,
            0x90, 0x90, 0xF0, 0x10, 0x10, 0xF0, 0x80, 0xF0, 0x10, 0xF0,
            0xF0, 0x80, 0xF0, 0x90, 0xF0, 0xF0, 0x10, 0x20, 0x40, 0x40,
            0xF0, 0x90, 0xF0, 0x90, 0xF0, 0xF0, 0x90, 0xF0, 0x10, 0xF0,
            0xF0, 0x90, 0xF0, 0x90, 0x90, 0xE0, 0x90, 0xE0, 0x90, 0xE0,
            0xF0, 0x80, 0x80, 0x80, 0xF0, 0xE0, 0x90, 0x90, 0x90, 0xE0,
            0xF0, 0x80, 0xF0, 0x80, 0xF0, 0xF0, 0x80, 0xF0, 0x80, 0x80
        };
        shared_font[threadIdx.x] = font_data[threadIdx.x];
    }
    __syncthreads();
    
    // Local state variables with register hints
    register unsigned short pc = program_counters[instance];
    register unsigned short index_reg = index_registers[instance];
    register unsigned char sp = stack_pointers[instance];
    register unsigned char dt = delay_timers[instance];
    register unsigned char st = sound_timers[instance];
    
    // Local counters
    register unsigned int local_instructions = 0;
    register unsigned int local_memory_ops = 0;
    register unsigned int local_array_reads = 0;
    register unsigned int local_array_writes = 0;
    register unsigned int local_comparisons = 0;
    register unsigned int local_swaps = 0;
    register unsigned int last_sort_check = 0;
    
    register bool array_was_accessed = false;
    register bool early_exit = false;
    
    // Skip crashed/halted instances early
    if (crashed[instance] || halted[instance]) {
        return;
    }
    
    // Optimized execution loop with reduced branching
    for (register int cycle = 0; cycle < cycles_to_run && !early_exit; cycle++) {
        // Bounds check with early exit
        if (__builtin_expect(pc >= 4094, 0)) {
            crashed[instance] = 1;
            break;
        }
        
        // Fetch instruction - coalesced memory access
        register unsigned short instruction = (memory[mem_base + pc] << 8) | memory[mem_base + pc + 1];
        register unsigned char opcode = instruction >> 12;  // Optimized shift
        register unsigned char x = (instruction >> 8) & 0xF;
        register unsigned char y = (instruction >> 4) & 0xF;
        register unsigned char n = instruction & 0xF;
        register unsigned char kk = instruction & 0xFF;
        register unsigned short nnn = instruction & 0xFFF;
        
        pc += 2;
        local_instructions++;
        
        // Optimized instruction execution with jump table simulation
        switch (opcode) {
            case 0x0:
                if (instruction == 0x00EE && sp > 0) {
                    sp--;
                    pc = stacks[stack_base + sp];
                }
                break;
                
            case 0x1: pc = nnn; break;
            
            case 0x2:
                if (__builtin_expect(sp < 16, 1)) {
                    stacks[stack_base + sp] = pc;
                    sp++;
                    pc = nnn;
                }
                break;
                
            case 0x3: 
                if (registers[reg_base + x] == kk) {
                    pc += 2;
                    local_comparisons++;
                }
                break;
                
            case 0x4: 
                if (registers[reg_base + x] != kk) {
                    pc += 2;
                    local_comparisons++;
                }
                break;
                
            case 0x5: 
                if (n == 0 && registers[reg_base + x] == registers[reg_base + y]) {
                    pc += 2;
                    local_comparisons++;
                }
                break;
                
            case 0x6: registers[reg_base + x] = kk; break;
            case 0x7: registers[reg_base + x] = (registers[reg_base + x] + kk) & 0xFF; break;
            
            case 0x8: // Register operations
                {
                    register unsigned char vx = registers[reg_base + x];
                    register unsigned char vy = registers[reg_base + y];
                    register unsigned char result = 0;
                    register unsigned char flag = 0;
                    
                    // Optimized switch with most common cases first
                    switch (n) {
                        case 0x0: 
                            result = vy;
                            local_swaps++;
                            break;
                        case 0x4: {
                            register int sum = vx + vy;
                            result = sum & 0xFF;
                            flag = (sum > 255) ? 1 : 0;
                        } break;
                        case 0x5: 
                            result = (vx - vy) & 0xFF; 
                            flag = (vx >= vy) ? 1 : 0;
                            local_comparisons++;
                            break;
                        case 0x7: 
                            result = (vy - vx) & 0xFF; 
                            flag = (vy >= vx) ? 1 : 0;
                            local_comparisons++;
                            break;
                        case 0x1: result = vx | vy; break;
                        case 0x2: result = vx & vy; break;
                        case 0x3: result = vx ^ vy; break;
                        case 0x6: result = vx >> 1; flag = vx & 0x1; break;
                        case 0xE: result = (vx << 1) & 0xFF; flag = (vx & 0x80) ? 1 : 0; break;
                        default: crashed[instance] = 1; continue;
                    }
                    
                    registers[reg_base + x] = result;
                    if (n != 0x0) {
                        registers[reg_base + 0xF] = flag;
                    }
                }
                break;
                
            case 0x9: 
                if (n == 0 && registers[reg_base + x] != registers[reg_base + y]) {
                    pc += 2;
                    local_comparisons++;
                }
                break;
                
            case 0xA: index_reg = nnn; break;
            
            case 0xB: 
                pc = nnn + registers[reg_base + (quirk_jumping ? ((nnn & 0xF00) >> 8) : 0)]; 
                break;
            
            case 0xC:
                // Optimized RNG with better distribution
                rng_state[instance] = rng_state[instance] * 1664525u + 1013904223u;
                registers[reg_base + x] = ((rng_state[instance] >> 16) & 0xFF) & kk;
                break;
                
            case 0xD: break; // Skip display
            case 0xE: break; // Skip key ops
                
            case 0xF: // Critical memory operations
                switch (kk) {
                    case 0x07: registers[reg_base + x] = dt; break;
                    case 0x15: dt = registers[reg_base + x]; break;
                    case 0x18: st = registers[reg_base + x]; break;
                    case 0x1E: index_reg = (index_reg + registers[reg_base + x]) & 0xFFFF; break;
                    case 0x29: index_reg = 0x50 + (registers[reg_base + x] & 0xF) * 5; break;
                    
                    case 0x33: // BCD
                        if (__builtin_expect(index_reg + 2 < 4096, 1)) {
                            register unsigned char value = registers[reg_base + x];
                            memory[mem_base + index_reg] = value / 100;
                            memory[mem_base + index_reg + 1] = (value / 10) % 10;
                            memory[mem_base + index_reg + 2] = value % 10;
                            
                            // Optimized sort array check
                            if (__builtin_expect(index_reg >= 0x300 && index_reg < 0x308, 0)) {
                                local_array_writes++;
                                array_was_accessed = true;
                            }
                        }
                        local_memory_ops++;
                        break;
                        
                    case 0x55: // Store registers - CRITICAL PATH
                        {
                            register int max_i = (index_reg + x < 4096) ? x : (4095 - index_reg);
                            for (register int i = 0; i <= max_i; i++) {
                                memory[mem_base + index_reg + i] = registers[reg_base + i];
                                
                                // Optimized array access detection
                                register unsigned short addr = index_reg + i;
                                if (__builtin_expect(addr >= 0x300 && addr < 0x308, 0)) {
                                    local_array_writes++;
                                    array_was_accessed = true;
                                }
                            }
                            if (quirk_memory) index_reg = (index_reg + x + 1) & 0xFFFF;
                        }
                        local_memory_ops++;
                        break;
                        
                    case 0x65: // Load memory - CRITICAL PATH
                        {
                            register int max_i = (index_reg + x < 4096) ? x : (4095 - index_reg);
                            for (register int i = 0; i <= max_i; i++) {
                                registers[reg_base + i] = memory[mem_base + index_reg + i];
                                
                                // Optimized array access detection
                                register unsigned short addr = index_reg + i;
                                if (__builtin_expect(addr >= 0x300 && addr < 0x308, 0)) {
                                    local_array_reads++;
                                    array_was_accessed = true;
                                }
                            }
                            if (quirk_memory) index_reg = (index_reg + x + 1) & 0xFFFF;
                        }
                        local_memory_ops++;
                        break;
                        
                    default: break;
                }
                break;
                
            default: crashed[instance] = 1; break;
        }
        
        // Optimized sorting check with reduced frequency for performance
        if (__builtin_expect(cycle > 1000 && sort_check_interval > 0 && 
            (cycle - last_sort_check) >= sort_check_interval && 
            array_was_accessed, 0)) {
            
            last_sort_check = cycle;
            
            // Load array into registers for faster access
            register unsigned char array[8];
            #pragma unroll
            for (register int i = 0; i < 8; i++) {
                array[i] = memory[mem_base + 0x300 + i];
            }
            
            // Optimized sorting detection
            register bool is_ascending = true;
            register bool is_descending = true;
            register bool has_variation = false;
            
            #pragma unroll
            for (register int i = 0; i < 7; i++) {
                register unsigned char curr = array[i];
                register unsigned char next = array[i + 1];
                
                if (curr > next) is_ascending = false;
                if (curr < next) is_descending = false;
                if (curr != next) has_variation = true;
            }
            
            // Detect valid sorting (must have variation, not all same values)
            if (__builtin_expect(has_variation && (is_ascending || is_descending), 0)) {
                sort_detected[instance] = 1;
                sort_direction[instance] = is_ascending ? 0 : 1;
                sort_cycle[instance] = cycle;
                early_exit = true; // Early exit optimization
            }
        }
        
        // Optimized timer updates
        if (__builtin_expect(cycle % timer_update_interval == 0, 0)) {
            if (dt > 0) dt--;
            if (st > 0) st--;
        }
    }
    
    // Write back state with coalesced access
    program_counters[instance] = pc;
    index_registers[instance] = index_reg;
    stack_pointers[instance] = sp;
    delay_timers[instance] = dt;
    sound_timers[instance] = st;
    
    // Write back statistics
    instructions_executed[instance] = local_instructions;
    memory_operations[instance] = local_memory_ops;
    array_accessed[instance] = array_was_accessed ? 1 : 0;
    array_reads[instance] = local_array_reads;
    array_writes[instance] = local_array_writes;
    comparison_operations[instance] = local_comparisons;
    swap_operations[instance] = local_swaps;
}
'''

class CUDASortingDetector:
    """
    Performance-optimized CUDA sorting algorithm detector
    Designed for maximum GPU utilization and throughput
    """
    
    def __init__(self, num_instances: int, quirks: dict = None):
        self.num_instances = num_instances
        
        # CHIP-8 quirks
        self.quirks = quirks or {
            'memory': True,
            'jumping': True, 
            'logic': True,
        }
        
        # Compile optimized CUDA kernel
        self.sorting_kernel = cp.RawKernel(CUDA_SORTING_KERNEL_OPTIMIZED, 'chip8_sorting_kernel_optimized')
        
        # Advanced GPU optimization
        current_device = cp.cuda.Device()
        props = cp.cuda.runtime.getDeviceProperties(current_device.id)
        gpu_name = props['name'].decode()
        gpu_memory_gb = props['totalGlobalMem'] / 1024**3
        max_threads_per_block = props['maxThreadsPerBlock']
        multiprocessor_count = props['multiProcessorCount']
        
        # Optimize block and grid sizes for maximum occupancy
        # Target: Use all SMs with maximum threads per block
        optimal_threads_per_block = min(1024, max_threads_per_block, num_instances)
        
        # For GTX 1070: 15 SMs, so aim for multiples that utilize all SMs
        threads_per_sm = optimal_threads_per_block
        total_optimal_threads = multiprocessor_count * threads_per_sm
        
        # Adjust based on memory constraints and instance count
        if gpu_memory_gb >= 8:
            self.block_size = 512  # Optimized for maximum GPU utilization
        elif gpu_memory_gb >= 6:
            self.block_size = 512  # Optimized for maximum GPU utilization
        else:
            self.block_size = 512  # Optimized for maximum GPU utilization
        
        # Ensure block size is optimal for the GPU architecture
        if 'GTX 1070' in gpu_name or 'GTX 1080' in gpu_name:
            self.block_size = 512  # Optimized for maximum GPU utilization  # Optimal for Pascal architecture
        elif 'RTX' in gpu_name:
            self.block_size = 512  # Optimized for maximum GPU utilization  # Optimal for Turing/Ampere
        
        self.grid_size = (num_instances + self.block_size - 1) // self.block_size
        
        # Calculate theoretical occupancy
        theoretical_occupancy = min(1.0, (self.grid_size * self.block_size) / total_optimal_threads)
        
        print(f"🚀 Optimized CUDA Sorting Detector: {num_instances:,} instances on {gpu_name}")
        print(f"   GPU: {multiprocessor_count} SMs, {max_threads_per_block} max threads/block")
        print(f"   Optimized: Block size {self.block_size}, Grid size {self.grid_size}")
        print(f"   Theoretical occupancy: {theoretical_occupancy:.1%}")
        print(f"   Total threads: {self.grid_size * self.block_size:,}")
        print(f"   Memory usage: {gpu_memory_gb:.1f} GB available")
        
        # Enable GPU optimizations
        cp.cuda.runtime.setDevice(current_device.id)
        
        self._initialize_state()
        self._initialize_sorting_detection()
        
        # Pre-warm GPU for better performance
        self._warmup_gpu()
    
    def _warmup_gpu(self):
        """Pre-warm GPU to reduce first-run overhead"""
        print("🔥 Warming up GPU...")
        dummy_array = cp.random.randint(0, 256, size=(1000, 1000), dtype=cp.uint8)
        _ = cp.sum(dummy_array)
        cp.cuda.Stream.null.synchronize()
        del dummy_array
        print("✅ GPU warmed up")
    
    def _initialize_state(self):
        """Initialize CHIP-8 state arrays with optimized memory layout"""
        print("🔧 Initializing optimized GPU memory layout...")
        
        # Use memory pools for better allocation performance
        mempool = cp.get_default_memory_pool()
        
        with cp.cuda.Device():
            # Standard CHIP-8 state with optimized alignment
            self.memory = cp.zeros((self.num_instances, 4096), dtype=cp.uint8)
            self.registers = cp.zeros((self.num_instances, 16), dtype=cp.uint8)
            self.index_register = cp.zeros(self.num_instances, dtype=cp.uint16)
            self.program_counter = cp.full(self.num_instances, 0x200, dtype=cp.uint16)
            self.stack_pointer = cp.zeros(self.num_instances, dtype=cp.uint8)
            self.stack = cp.zeros((self.num_instances, 16), dtype=cp.uint16)
            self.delay_timer = cp.zeros(self.num_instances, dtype=cp.uint8)
            self.sound_timer = cp.zeros(self.num_instances, dtype=cp.uint8)
            self.crashed = cp.zeros(self.num_instances, dtype=cp.uint8)
            self.halted = cp.zeros(self.num_instances, dtype=cp.uint8)
            
            # Optimized RNG initialization
            self.rng_state = cp.random.randint(1, 2**32-1, size=self.num_instances, dtype=cp.uint32)
            
            # Load CHIP-8 font with optimized access pattern
            font = cp.array([
                0xF0, 0x90, 0x90, 0x90, 0xF0, 0x20, 0x60, 0x20, 0x20, 0x70,
                0xF0, 0x10, 0xF0, 0x80, 0xF0, 0xF0, 0x10, 0xF0, 0x10, 0xF0,
                0x90, 0x90, 0xF0, 0x10, 0x10, 0xF0, 0x80, 0xF0, 0x10, 0xF0,
                0xF0, 0x80, 0xF0, 0x90, 0xF0, 0xF0, 0x10, 0x20, 0x40, 0x40,
                0xF0, 0x90, 0xF0, 0x90, 0xF0, 0xF0, 0x90, 0xF0, 0x10, 0xF0,
                0xF0, 0x90, 0xF0, 0x90, 0x90, 0xE0, 0x90, 0xE0, 0x90, 0xE0,
                0xF0, 0x80, 0x80, 0x80, 0xF0, 0xE0, 0x90, 0x90, 0x90, 0xE0,
                0xF0, 0x80, 0xF0, 0x80, 0xF0, 0xF0, 0x80, 0xF0, 0x80, 0x80
            ], dtype=cp.uint8)
            
            # Broadcast font to all instances efficiently
            font_broadcast = cp.broadcast_to(font, (self.num_instances, len(font)))
            self.memory[:, 0x50:0x50 + len(font)] = font_broadcast
            
            # Statistics
            self.stats = {
                'instructions_executed': cp.zeros(self.num_instances, dtype=cp.uint32),
                'memory_operations': cp.zeros(self.num_instances, dtype=cp.uint32),
            }
        
        print(f"📊 Memory allocated: {mempool.used_bytes() / 1024**3:.2f} GB")
    
    def _initialize_sorting_detection(self):
        """Initialize sorting detection arrays"""
        self.sort_detected = cp.zeros(self.num_instances, dtype=cp.uint8)
        self.sort_direction = cp.zeros(self.num_instances, dtype=cp.uint8)
        self.sort_cycle = cp.zeros(self.num_instances, dtype=cp.uint32)
        self.array_accessed = cp.zeros(self.num_instances, dtype=cp.uint8)
        self.array_reads = cp.zeros(self.num_instances, dtype=cp.uint32)
        self.array_writes = cp.zeros(self.num_instances, dtype=cp.uint32)
        self.comparison_operations = cp.zeros(self.num_instances, dtype=cp.uint32)
        self.swap_operations = cp.zeros(self.num_instances, dtype=cp.uint32)
    
    def load_roms_with_sort_arrays(self, rom_data_list: List[Union[bytes, np.ndarray]]):
        """Load ROMs and pre-seed sorting arrays with optimized GPU operations"""
        print(f"🚀 Loading {len(rom_data_list)} ROMs with optimized GPU operations...")
        
        # Use GPU streams for parallel operations
        stream = cp.cuda.Stream()
        
        with stream:
            # Reset state for new batch
            self.program_counter.fill(0x200)
            self.crashed.fill(0)
            self.halted.fill(0)
            self.sort_detected.fill(0)
            self.array_accessed.fill(0)
            
            # Load ROMs in batches for better memory usage
            batch_size = min(1000, len(rom_data_list))
            
            for batch_start in range(0, self.num_instances, batch_size):
                batch_end = min(batch_start + batch_size, self.num_instances)
                current_batch_size = batch_end - batch_start
                
                # Prepare ROM batch
                rom_batch = []
                for i in range(current_batch_size):
                    rom_idx = (batch_start + i) % len(rom_data_list)
                    rom_data = rom_data_list[rom_idx]
                    
                    if isinstance(rom_data, np.ndarray):
                        rom_bytes = rom_data
                    else:
                        rom_bytes = np.frombuffer(rom_data, dtype=np.uint8)
                    
                    if len(rom_bytes) > 4096 - 0x200:
                        rom_bytes = rom_bytes[:4096 - 0x200]
                    
                    rom_batch.append(rom_bytes)
                
                # Load ROM batch to GPU efficiently
                max_rom_size = max(len(rom) for rom in rom_batch)
                rom_array = np.zeros((current_batch_size, max_rom_size), dtype=np.uint8)
                
                for i, rom in enumerate(rom_batch):
                    rom_array[i, :len(rom)] = rom
                
                # Transfer to GPU and load into memory
                rom_gpu = cp.array(rom_array)
                for i in range(current_batch_size):
                    instance_idx = batch_start + i
                    rom_size = len(rom_batch[i])
                    self.memory[instance_idx, 0x200:0x200 + rom_size] = rom_gpu[i, :rom_size]
                
                # Generate optimized sorting arrays for this batch
                random_arrays = cp.random.randint(1, 256, size=(current_batch_size, SORT_ARRAY_SIZE), dtype=cp.uint8)
                
                # Shuffle each array to ensure it's not pre-sorted
                for i in range(current_batch_size):
                    shuffled = cp.random.permutation(random_arrays[i])
                    instance_idx = batch_start + i
                    self.memory[instance_idx, SORT_ARRAY_START:SORT_ARRAY_START + SORT_ARRAY_SIZE] = shuffled
        
        # Ensure all operations complete
        stream.synchronize()
        print(f"✅ Loaded {self.num_instances:,} ROMs with optimized sort arrays")
    
    def run_sorting_detection(self, cycles: int = 50000, sort_check_interval: int = SORT_CHECK_INTERVAL):
        """Run optimized CUDA sorting detection with maximum GPU utilization"""
        print(f"🚀 Launching OPTIMIZED CUDA sorting detection...")
        print(f"   Cycles: {cycles:,}")
        print(f"   Block size: {self.block_size} (optimized)")
        print(f"   Grid size: {self.grid_size}")
        print(f"   Total GPU threads: {self.grid_size * self.block_size:,}")
        
        # Create high-priority CUDA stream
        stream = cp.cuda.Stream(non_blocking=True)
        
        start_time = time.time()
        
        with stream:
            # Launch optimized kernel with maximum occupancy
            self.sorting_kernel(
                (self.grid_size,), (self.block_size,),
                (
                    # CHIP-8 state
                    self.memory, self.registers, self.index_register, self.program_counter,
                    self.stack_pointer, self.stack, self.delay_timer, self.sound_timer,
                    self.crashed, self.halted, self.rng_state,
                    
                    # Sorting detection results
                    self.sort_detected, self.sort_direction, self.sort_cycle,
                    self.array_accessed, self.array_reads, self.array_writes,
                    self.comparison_operations, self.swap_operations,
                    
                    # Statistics
                    self.stats['instructions_executed'], self.stats['memory_operations'],
                    
                    # Parameters
                    self.num_instances, cycles, 16, sort_check_interval,
                    
                    # Quirks
                    1 if self.quirks['memory'] else 0,
                    1 if self.quirks['jumping'] else 0,
                    1 if self.quirks['logic'] else 0
                ),
                stream=stream
            )
        
        # Force synchronization and measure actual GPU time
        stream.synchronize()
        execution_time = time.time() - start_time
        
        # Report optimized results
        total_instructions = int(cp.sum(self.stats['instructions_executed']))
        sorts_found = int(cp.sum(self.sort_detected))
        arrays_accessed = int(cp.sum(self.array_accessed))
        
        # Calculate performance metrics
        total_operations = self.num_instances * cycles
        operations_per_second = total_operations / execution_time if execution_time > 0 else 0
        roms_per_second = self.num_instances / execution_time if execution_time > 0 else 0
        
        print(f"🚀 OPTIMIZED detection complete: {execution_time:.2f}s")
        print(f"   Instructions executed: {total_instructions:,}")
        print(f"   Operations/second: {operations_per_second:,.0f}")
        print(f"   ROMs/second: {roms_per_second:,.0f}")
        print(f"   ROMs accessing sort array: {arrays_accessed}")
        print(f"   🎯 SORTING ALGORITHMS FOUND: {sorts_found}")
        
        if sorts_found > 0:
            self._report_sorting_discoveries()
        
        return sorts_found
    
    def get_gpu_utilization_info(self):
        """Get detailed GPU utilization information"""
        current_device = cp.cuda.Device()
        props = cp.cuda.runtime.getDeviceProperties(current_device.id)
        
        # Calculate theoretical peak utilization
        max_threads = props['multiProcessorCount'] * props['maxThreadsPerMultiProcessor']
        actual_threads = self.grid_size * self.block_size
        utilization = min(100.0, (actual_threads / max_threads) * 100)
        
        return {
            'gpu_name': props['name'].decode(),
            'multiprocessor_count': props['multiProcessorCount'],
            'max_threads_per_block': props['maxThreadsPerBlock'],
            'max_threads_total': max_threads,
            'actual_threads': actual_threads,
            'theoretical_utilization': utilization,
            'block_size': self.block_size,
            'grid_size': self.grid_size
        }
    
    def _report_sorting_discoveries(self):
        """Report discovered sorting algorithms with enhanced details"""
        sort_mask = cp.asnumpy(self.sort_detected).astype(bool)
        sort_directions = cp.asnumpy(self.sort_direction)
        sort_cycles = cp.asnumpy(self.sort_cycle)
        array_reads_np = cp.asnumpy(self.array_reads)
        array_writes_np = cp.asnumpy(self.array_writes)
        comparisons_np = cp.asnumpy(self.comparison_operations)
        swaps_np = cp.asnumpy(self.swap_operations)
        
        print(f"\n🎯 SORTING ALGORITHM DISCOVERIES:")
        print("=" * 60)
        
        discoveries = []
        for i in range(self.num_instances):
            if sort_mask[i]:
                direction = "ASCENDING" if sort_directions[i] == 0 else "DESCENDING"
                cycle = sort_cycles[i]
                reads = array_reads_np[i]
                writes = array_writes_np[i]
                comps = comparisons_np[i]
                swaps = swaps_np[i]
                
                discoveries.append({
                    'instance': i,
                    'direction': direction,
                    'cycle': cycle,
                    'reads': reads,
                    'writes': writes,
                    'comparisons': comps,
                    'swaps': swaps
                })
                
                # Calculate efficiency score
                total_ops = reads + writes + comps + swaps
                if total_ops > 0:
                    efficiency = max(0, 100 - (total_ops - 16) * 2)  # Penalty for excessive operations
                else:
                    efficiency = 0
                
                print(f"🔢 Instance {i:05d}: {direction} sort at cycle {cycle:,}")
                print(f"   Operations: R:{reads} W:{writes} C:{comps} S:{swaps} (Total:{total_ops})")
                print(f"   Efficiency score: {efficiency:.0f}/100")
                
                # Show the sorted array
                sorted_array = cp.asnumpy(self.memory[i, SORT_ARRAY_START:SORT_ARRAY_START + SORT_ARRAY_SIZE])
                print(f"   Final array: {list(sorted_array)}")
                print()
        
        return discoveries
    
    def get_sorting_results(self) -> Dict:
        """Get comprehensive sorting detection results"""
        sort_mask = cp.asnumpy(self.sort_detected).astype(bool)
        
        results = {
            'sorts_found': int(cp.sum(self.sort_detected)),
            'arrays_accessed': int(cp.sum(self.array_accessed)),
            'total_array_reads': int(cp.sum(self.array_reads)),
            'total_array_writes': int(cp.sum(self.array_writes)),
            'total_comparisons': int(cp.sum(self.comparison_operations)),
            'total_swaps': int(cp.sum(self.swap_operations)),
            'gpu_utilization': self.get_gpu_utilization_info(),
            'discoveries': []
        }
        
        for i in range(self.num_instances):
            if sort_mask[i]:
                results['discoveries'].append({
                    'instance_id': i,
                    'sort_direction': 'ascending' if cp.asnumpy(self.sort_direction)[i] == 0 else 'descending',
                    'sort_cycle': int(cp.asnumpy(self.sort_cycle)[i]),
                    'array_reads': int(cp.asnumpy(self.array_reads)[i]),
                    'array_writes': int(cp.asnumpy(self.array_writes)[i]),
                    'comparisons': int(cp.asnumpy(self.comparison_operations)[i]),
                    'swaps': int(cp.asnumpy(self.swap_operations)[i]),
                    'final_array': cp.asnumpy(self.memory[i, SORT_ARRAY_START:SORT_ARRAY_START + SORT_ARRAY_SIZE]).tolist()
                })
        
        return results


class PerformanceOptimizedROMGenerator:
    """
    Maximum performance ROM generator for sustained GPU utilization
    """
    
    def __init__(self, rom_size: int = 3584):
        self.rom_size = rom_size
        
        # Get GPU info and optimize
        current_device = cp.cuda.Device()
        props = cp.cuda.runtime.getDeviceProperties(current_device.id)
        gpu_name = props['name'].decode()
        gpu_memory_gb = props['totalGlobalMem'] / 1024**3
        
        print(f"🚀 Performance-Optimized ROM Generator on {gpu_name}")
        print(f"   Memory: {gpu_memory_gb:.1f} GB")
        print(f"   ROM size: {rom_size} bytes")
        
        # Pre-allocate memory pool for sustained performance
        self._warmup_generator()
    
    def _warmup_generator(self):
        """Pre-warm generator for sustained performance"""
        print("🔥 Pre-warming ROM generator...")
        
        # Pre-allocate and deallocate to warm up memory allocator
        for size in [1000, 10000, 50000]:
            dummy = cp.random.randint(0, 256, size=(size, self.rom_size), dtype=cp.uint8)
            del dummy
        
        cp.cuda.Stream.null.synchronize()
        print("✅ ROM generator warmed up")
    
    def generate_batch_optimized(self, count: int) -> List[np.ndarray]:
        """Generate ROM batch with maximum GPU utilization"""
        start_time = time.time()
        
        # Use high-priority stream
        stream = cp.cuda.Stream(non_blocking=True)
        
        with stream:
            # Generate all ROMs in single GPU operation
            all_roms_gpu = cp.random.randint(
                0, 256, 
                size=(count, self.rom_size), 
                dtype=cp.uint8
            )
            
            # Transfer to CPU in optimal chunks to avoid memory pressure
            chunk_size = min(10000, count)
            rom_list = []
            
            for i in range(0, count, chunk_size):
                end_idx = min(i + chunk_size, count)
                chunk = cp.asnumpy(all_roms_gpu[i:end_idx])
                
                for j in range(chunk.shape[0]):
                    rom_list.append(chunk[j])
        
        # Ensure completion
        stream.synchronize()
        
        generation_time = time.time() - start_time
        roms_per_second = count / generation_time if generation_time > 0 else 0
        
        print(f"🚀 Generated {count:,} ROMs in {generation_time:.2f}s ({roms_per_second:,.0f} ROMs/sec)")
        
        return rom_list


def run_optimized_search(batch_size: int = 50000, cycles: int = 50000, 
                        continuous: bool = True, check_interval: int = 500):
    """
    Run optimized sorting search with maximum GPU utilization
    """
    print("🚀 MAXIMUM PERFORMANCE SORTING SEARCH")
    print("=" * 60)
    print(f"Batch size: {batch_size:,} ROMs (optimized for max GPU usage)")
    print(f"Cycles: {cycles:,}")
    print(f"Check interval: {check_interval}")
    print(f"Mode: {'Continuous' if continuous else 'Single batch'}")
    print()
    
    # Initialize optimized components
    detector = CUDASortingDetector(batch_size)
    generator = PerformanceOptimizedROMGenerator()
    
    # Display GPU utilization info
    util_info = detector.get_gpu_utilization_info()
    print(f"🔧 GPU Configuration:")
    print(f"   GPU: {util_info['gpu_name']}")
    print(f"   SMs: {util_info['multiprocessor_count']}")
    print(f"   Max threads: {util_info['max_threads_total']:,}")
    print(f"   Using threads: {util_info['actual_threads']:,}")
    print(f"   Theoretical utilization: {util_info['theoretical_utilization']:.1f}%")
    print()
    
    total_roms = 0
    total_sorts = 0
    batch_count = 0
    start_time = time.time()
    
    try:
        while True:
            batch_count += 1
            print(f"🚀 Optimized Batch {batch_count}")
            print("-" * 40)
            
            # Generate ROMs with maximum performance
            rom_data_list = generator.generate_batch_optimized(batch_size)
            
            # Load and run with optimal GPU utilization
            detector.load_roms_with_sort_arrays(rom_data_list)
            sorts_found = detector.run_sorting_detection(cycles=cycles, sort_check_interval=check_interval)
            
            # Update statistics
            total_roms += batch_size
            total_sorts += sorts_found
            
            # Performance report
            elapsed = time.time() - start_time
            rate = total_roms / elapsed if elapsed > 0 else 0
            
            print(f"\n📊 Performance Report:")
            print(f"   Batches: {batch_count}")
            print(f"   Total ROMs: {total_roms:,}")
            print(f"   Rate: {rate:,.0f} ROMs/sec")
            print(f"   Runtime: {elapsed/3600:.2f} hours")
            print(f"   🎯 Sorts found: {total_sorts}")
            
            if total_sorts > 0:
                discovery_rate = total_roms // total_sorts
                print(f"   Discovery rate: 1 per {discovery_rate:,} ROMs")
            
            print("=" * 60)
            
            if not continuous:
                break
                
    except KeyboardInterrupt:
        print(f"\n🛑 Search interrupted by user")
    
    # Final summary
    final_time = time.time() - start_time
    final_rate = total_roms / final_time if final_time > 0 else 0
    
    print(f"\n🏁 OPTIMIZED SEARCH COMPLETE")
    print("=" * 60)
    print(f"Total ROMs processed: {total_roms:,}")
    print(f"Total time: {final_time/3600:.2f} hours")
    print(f"Final rate: {final_rate:,.0f} ROMs/sec")
    print(f"🎯 TOTAL SORTING ALGORITHMS: {total_sorts}")
    
    return total_sorts


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Optimized CUDA Sorting Algorithm Detector')
    parser.add_argument('--batch-size', type=int, default=50000,
                       help='ROMs per batch - optimized for max GPU usage (default: 50000)')
    parser.add_argument('--cycles', type=int, default=50000,
                       help='Execution cycles per ROM (default: 50000)')
    parser.add_argument('--continuous', action='store_true', default=True,
                       help='Run continuous search (default: True)')
    parser.add_argument('--check-interval', type=int, default=500,
                       help='Sort check interval (default: 500)')
    
    args = parser.parse_args()
    
    run_optimized_search(
        batch_size=args.batch_size,
        cycles=args.cycles,
        continuous=args.continuous,
        check_interval=args.check_interval
    )