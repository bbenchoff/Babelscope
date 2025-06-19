"""
CHIP-8 Emulator for Babelscope
The 'toy' version for the Bablescope.
Does not support massive parallelism or advanced features.
Focuses purely on emulation - ROM generation and analysis are handled elsewhere.
But hey you can use this to test ROMs or something.
"""

import cupy as cp
import numpy as np
from typing import Dict, Tuple, Optional, List, Union
import time
import tkinter as tk
from tkinter import Canvas
import logging
import os

# CHIP-8 System Constants
MEMORY_SIZE = 4096
DISPLAY_WIDTH = 64
DISPLAY_HEIGHT = 32
DISPLAY_PIXELS = DISPLAY_WIDTH * DISPLAY_HEIGHT
REGISTER_COUNT = 16
STACK_SIZE = 16
KEYPAD_SIZE = 16
PROGRAM_START = 0x200
FONT_START = 0x50
FONT_SIZE = 80

# CHIP-8 Font set (hexadecimal digits 0-F)
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

class Chip8Emulator:
    """
    Single-instance CHIP-8 emulator with display output capability.
    Can be used for testing known ROMs or analyzing individual instances.
    """
    
    def __init__(self, debug_file: str = None, quirks: dict = None):
        self.debug_file = debug_file
        self.debug_log = []
        
        # CHIP-8 Quirks configuration
        self.quirks = quirks or {
            'memory': True,      # Fx55/Fx65 increment I register  
            'display_wait': False, # Drawing waits for vblank (60fps limit) - DISABLED by default
            'jumping': True,     # Bnnn uses vX instead of v0
            'shifting': False,   # 8xy6/8xyE use vY or vX (False = use vX)
            'logic': True,       # 8xy1/8xy2/8xy3 reset vF to 0
        }
        
        # Display wait tracking
        self.last_draw_time = 0
        self.vblank_wait = False
        
        self.reset()
    
    def log_debug(self, message: str):
        """Log debug message to both console and file"""
        print(message)
        self.debug_log.append(message)
        
        if self.debug_file:
            with open(self.debug_file, 'a', encoding='utf-8') as f:
                f.write(message + '\n')
    
    def reset(self):
        """Reset the emulator to initial state"""
        self.memory = np.zeros(MEMORY_SIZE, dtype=np.uint8)
        self.display = np.zeros((DISPLAY_HEIGHT, DISPLAY_WIDTH), dtype=np.uint8)
        self.registers = np.zeros(REGISTER_COUNT, dtype=np.uint8)
        # FIXED: Use regular Python int for index_register to avoid overflow
        self.index_register = 0  # This should be a regular int, not uint8
        self.program_counter = PROGRAM_START
        self.stack_pointer = 0
        self.stack = np.zeros(STACK_SIZE, dtype=np.uint16)
        self.delay_timer = 0
        self.sound_timer = 0
        self.keypad = np.zeros(KEYPAD_SIZE, dtype=np.uint8)
        self.waiting_for_key = False
        self.key_register = 0
        
        # Load font into memory
        self.memory[FONT_START:FONT_START + FONT_SIZE] = cp.asnumpy(CHIP8_FONT)
        
        # Instrumentation
        self.stats = {
            'instructions_executed': 0,
            'display_writes': 0,
            'display_clears': 0,
            'sprite_collisions': 0,
            'memory_reads': 0,
            'memory_writes': 0,
            'timer_sets': 0,
            'sound_activations': 0,
            'key_checks': 0,
            'blocking_key_waits': 0,
            'jumps_taken': 0,
            'subroutine_calls': 0,
            'returns': 0,
            'stack_operations': 0,
            'register_operations': 0,
            'arithmetic_operations': 0,
            'logical_operations': 0,
            'random_generations': 0,
            'pixels_drawn': 0,
            'pixels_erased': 0,
            'unique_pixels_touched': 0,
            'display_updates': 0,
            'cycles_executed': 0
        }
        
        self.pixel_touched = np.zeros((DISPLAY_HEIGHT, DISPLAY_WIDTH), dtype=bool)
        self.crashed = False
        self.halt = False
    
    def load_rom(self, rom_data: Union[bytes, np.ndarray, str]):
        """Load a ROM into memory"""
        if isinstance(rom_data, str):
            with open(rom_data, 'rb') as f:
                rom_bytes = f.read()
        elif isinstance(rom_data, np.ndarray):
            rom_bytes = rom_data.tobytes()
        else:
            rom_bytes = rom_data
        
        if len(rom_bytes) > MEMORY_SIZE - PROGRAM_START:
            raise ValueError(f"ROM too large: {len(rom_bytes)} bytes, max {MEMORY_SIZE - PROGRAM_START}")
        
        # Load ROM starting at 0x200
        for i, byte in enumerate(rom_bytes):
            self.memory[PROGRAM_START + i] = byte
            
        print(f"Loaded ROM: {len(rom_bytes)} bytes")
        print(f"First instruction: 0x{self.memory[PROGRAM_START]:02X}{self.memory[PROGRAM_START+1]:02X}")
    
    def step(self):
        """Execute one instruction"""
        if self.crashed or self.halt:
            return False
        
        if self.waiting_for_key:
            self.stats['blocking_key_waits'] += 1
            return True
        
        # Fetch instruction
        if self.program_counter >= MEMORY_SIZE - 1:
            self.crashed = True
            return False
        
        # Convert to regular Python int to avoid numpy overflow issues
        high_byte = int(self.memory[self.program_counter])
        low_byte = int(self.memory[self.program_counter + 1])
        instruction = (high_byte << 8) | low_byte
        
        self.program_counter += 2
        self.stats['instructions_executed'] += 1
        
        # Decode and execute
        return self._execute_instruction(instruction)
    
    def run_interactive(self, show_display: bool = True, scale: int = 8, title: str = "CHIP-8 Interactive"):
        """Run the emulator in interactive mode with real-time input and display"""
        if show_display:
            self.show_display(scale=scale, title=title)
        else:
            # Non-GUI mode - just run continuously
            import time
            while not self.crashed and not self.halt:
                start_time = time.time()
                
                # Run multiple cycles per frame for better performance
                for _ in range(10):
                    if not self.step():
                        break
                
                # Update timers at ~60Hz
                if self.delay_timer > 0:
                    self.delay_timer -= 1
                if self.sound_timer > 0:
                    self.sound_timer -= 1
                
                # Maintain roughly 60 FPS timing
                elapsed = time.time() - start_time
                sleep_time = max(0, 1/60 - elapsed)
                time.sleep(sleep_time)
    
    def run(self, max_cycles: int = 1000, update_timers: bool = True):
        """Run emulator for specified number of cycles"""
        for cycle in range(max_cycles):
            if not self.step():
                break
            
            self.stats['cycles_executed'] += 1
            
            # Update timers at ~60Hz (every 16-17 cycles assuming 1000Hz execution)
            if update_timers and cycle % 16 == 0:
                if self.delay_timer > 0:
                    self.delay_timer -= 1
                if self.sound_timer > 0:
                    self.sound_timer -= 1
    
    def _execute_instruction(self, instruction: int) -> bool:
        """Execute a single CHIP-8 instruction"""
        # Ensure instruction is a regular Python int, not numpy int
        instruction = int(instruction)
        
        # Decode instruction
        opcode = (instruction & 0xF000) >> 12
        x = (instruction & 0x0F00) >> 8
        y = (instruction & 0x00F0) >> 4
        n = instruction & 0x000F
        kk = instruction & 0x00FF
        nnn = instruction & 0x0FFF
        
        # Debug print for failing tests
        if self.stats['instructions_executed'] < 20 or instruction == 0x0000:
            msg = f"Executing: 0x{instruction:04X} at PC=0x{self.program_counter-2:03X}"
            self.log_debug(msg)
            msg2 = f"  Opcode: 0x{opcode:X}, x={x}, y={y}, n={n}, kk=0x{kk:02X}, nnn=0x{nnn:03X}"
            self.log_debug(msg2)
            
            # Also log register state
            if self.stats['instructions_executed'] < 5:
                reg_state = f"  Registers: V0-V7={[int(self.registers[i]) for i in range(8)]}"
                self.log_debug(reg_state)
                reg_state2 = f"            V8-VF={[int(self.registers[i]) for i in range(8, 16)]}"
                self.log_debug(reg_state2)
                idx_state = f"  I=0x{self.index_register:03X}, SP={self.stack_pointer}"
                self.log_debug(idx_state)
        
        try:
            if instruction == 0x00E0:  # CLS
                self.display.fill(0)
                self.pixel_touched.fill(False)
                self.stats['display_clears'] += 1
                self.stats['display_updates'] += 1
                
            elif instruction == 0x00EE:  # RET
                if self.stack_pointer > 0:
                    self.stack_pointer -= 1
                    self.program_counter = int(self.stack[self.stack_pointer])
                    self.stats['returns'] += 1
                    self.stats['stack_operations'] += 1
                else:
                    self.log_debug(f"ERROR: RET with empty stack at PC=0x{self.program_counter-2:03X}")
                    self.crashed = True
                    return False
            
            elif opcode == 0x0:  # SYS addr (should be ignored in modern interpreters)
                # Most modern interpreters ignore SYS instructions
                pass
                    
            elif opcode == 0x1:  # JP addr
                if nnn == self.program_counter - 2:
                    # Only log infinite jump warning once, then set a flag to suppress further warnings
                    if not hasattr(self, '_infinite_jump_warned'):
                        self.log_debug(f"WARNING: Infinite jump detected at PC=0x{self.program_counter-2:03X} (further warnings suppressed)")
                        self._infinite_jump_warned = True
                self.program_counter = nnn
                self.stats['jumps_taken'] += 1
                
            elif opcode == 0x2:  # CALL addr
                if self.stack_pointer >= STACK_SIZE:
                    self.log_debug(f"ERROR: Stack overflow at PC=0x{self.program_counter-2:03X}")
                    self.crashed = True
                    return False
                self.stack[self.stack_pointer] = self.program_counter
                self.stack_pointer += 1
                self.program_counter = nnn
                self.stats['subroutine_calls'] += 1
                self.stats['stack_operations'] += 1
                    
            elif opcode == 0x3:  # SE Vx, byte
                if int(self.registers[x]) == kk:
                    self.program_counter += 2
                self.stats['register_operations'] += 1
                
            elif opcode == 0x4:  # SNE Vx, byte
                if int(self.registers[x]) != kk:
                    self.program_counter += 2
                self.stats['register_operations'] += 1
                
            elif opcode == 0x5:  # SE Vx, Vy
                if n != 0:  # 5xy0 format check
                    self.log_debug(f"ERROR: Invalid 5xy{n:X} instruction at PC=0x{self.program_counter-2:03X}")
                    self.crashed = True
                    return False
                if int(self.registers[x]) == int(self.registers[y]):
                    self.program_counter += 2
                self.stats['register_operations'] += 1
                
            elif opcode == 0x6:  # LD Vx, byte
                self.registers[x] = kk
                self.stats['register_operations'] += 1
                
            elif opcode == 0x7:  # ADD Vx, byte
                self.registers[x] = (int(self.registers[x]) + kk) & 0xFF
                self.stats['arithmetic_operations'] += 1
                self.stats['register_operations'] += 1
                
            elif opcode == 0x8:  # Register operations
                self.stats['register_operations'] += 1
                if n == 0x0:  # LD Vx, Vy
                    self.registers[x] = self.registers[y]
                elif n == 0x1:  # OR Vx, Vy
                    self.registers[x] = int(self.registers[x]) | int(self.registers[y])
                    self.registers[0xF] = 0  # VF should be reset
                    self.stats['logical_operations'] += 1
                elif n == 0x2:  # AND Vx, Vy
                    self.registers[x] = int(self.registers[x]) & int(self.registers[y])
                    self.registers[0xF] = 0  # VF should be reset
                    self.stats['logical_operations'] += 1
                elif n == 0x3:  # XOR Vx, Vy
                    self.registers[x] = int(self.registers[x]) ^ int(self.registers[y])
                    self.registers[0xF] = 0  # VF should be reset
                    self.stats['logical_operations'] += 1
                elif n == 0x4:  # ADD Vx, Vy
                    # CRITICAL: Store operands BEFORE modifying VF
                    vx_val = int(self.registers[x])
                    vy_val = int(self.registers[y])
                    result = vx_val + vy_val
                    self.registers[x] = result & 0xFF
                    self.registers[0xF] = 1 if result > 255 else 0
                    self.stats['arithmetic_operations'] += 1
                elif n == 0x5:  # SUB Vx, Vy
                    # CRITICAL: Store operands BEFORE modifying VF
                    vx_val = int(self.registers[x])
                    vy_val = int(self.registers[y])
                    self.registers[x] = (vx_val - vy_val) & 0xFF
                    self.registers[0xF] = 1 if vx_val >= vy_val else 0  # NOT borrow
                    self.stats['arithmetic_operations'] += 1
                elif n == 0x6:  # SHR Vx {, Vy}
                    # CRITICAL: Store operand BEFORE modifying VF
                    vx_val = int(self.registers[x])
                    self.registers[x] = vx_val >> 1
                    self.registers[0xF] = vx_val & 0x1  # Shifted out bit
                    self.stats['logical_operations'] += 1
                elif n == 0x7:  # SUBN Vx, Vy
                    # CRITICAL: Store operands BEFORE modifying VF
                    vx_val = int(self.registers[x])
                    vy_val = int(self.registers[y])
                    self.registers[x] = (vy_val - vx_val) & 0xFF
                    self.registers[0xF] = 1 if vy_val >= vx_val else 0  # NOT borrow
                    self.stats['arithmetic_operations'] += 1
                elif n == 0xE:  # SHL Vx {, Vy}
                    # CRITICAL: Store operand BEFORE modifying VF
                    vx_val = int(self.registers[x])
                    self.registers[x] = (vx_val << 1) & 0xFF
                    self.registers[0xF] = 1 if (vx_val & 0x80) else 0  # Shifted out bit
                    self.stats['logical_operations'] += 1
                else:
                    self.log_debug(f"ERROR: Unknown 8xy{n:X} instruction at PC=0x{self.program_counter-2:03X}")
                    self.crashed = True
                    return False
                    
            elif opcode == 0x9:  # SNE Vx, Vy
                if n != 0:  # 9xy0 format check
                    self.log_debug(f"ERROR: Invalid 9xy{n:X} instruction at PC=0x{self.program_counter-2:03X}")
                    self.crashed = True
                    return False
                if int(self.registers[x]) != int(self.registers[y]):
                    self.program_counter += 2
                self.stats['register_operations'] += 1
                
            elif opcode == 0xA:  # LD I, addr
                self.index_register = nnn
                self.stats['register_operations'] += 1
                
            elif opcode == 0xB:  # JP V0, addr (with jumping quirk)
                if self.quirks['jumping']:
                    # Modern quirk: use vX where X is the high nibble of nnn
                    x = (nnn & 0xF00) >> 8
                    self.program_counter = nnn + int(self.registers[x])
                else:
                    # Classic behavior: always use v0
                    self.program_counter = nnn + int(self.registers[0])
                self.stats['jumps_taken'] += 1
                
            elif opcode == 0xC:  # RND Vx, byte
                random_byte = np.random.randint(0, 256)
                self.registers[x] = random_byte & kk
                self.stats['random_generations'] += 1
                self.stats['register_operations'] += 1
                
            elif opcode == 0xD:  # DRW Vx, Vy, nibble
                self._draw_sprite(x, y, n)
                
            elif opcode == 0xE:
                self.stats['key_checks'] += 1
                if kk == 0x9E:  # SKP Vx
                    key_val = int(self.registers[x]) & 0xF
                    if self.keypad[key_val]:
                        self.program_counter += 2
                elif kk == 0xA1:  # SKNP Vx
                    key_val = int(self.registers[x]) & 0xF
                    if not self.keypad[key_val]:
                        self.program_counter += 2
                else:
                    self.log_debug(f"ERROR: Unknown Ex{kk:02X} instruction at PC=0x{self.program_counter-2:03X}")
                    self.crashed = True
                    return False
                        
            elif opcode == 0xF:
                if kk == 0x07:  # LD Vx, DT
                    self.registers[x] = self.delay_timer
                    self.stats['register_operations'] += 1
                elif kk == 0x0A:  # LD Vx, K
                    self.waiting_for_key = True
                    self.key_register = x
                    self.stats['blocking_key_waits'] += 1
                elif kk == 0x15:  # LD DT, Vx
                    self.delay_timer = int(self.registers[x])
                    self.stats['timer_sets'] += 1
                elif kk == 0x18:  # LD ST, Vx
                    self.sound_timer = int(self.registers[x])
                    self.stats['timer_sets'] += 1
                    if int(self.registers[x]) > 0:
                        self.stats['sound_activations'] += 1
                elif kk == 0x1E:  # ADD I, Vx
                    # FIXED: Proper handling of index register addition with bounds checking
                    new_value = self.index_register + int(self.registers[x])
                    self.index_register = new_value & 0xFFFF  # Keep within 16-bit range
                    self.stats['arithmetic_operations'] += 1
                elif kk == 0x29:  # LD F, Vx
                    digit = int(self.registers[x]) & 0xF
                    self.index_register = FONT_START + digit * 5
                elif kk == 0x33:  # LD B, Vx
                    value = int(self.registers[x])
                    if self.index_register + 2 < MEMORY_SIZE:
                        self.memory[self.index_register] = value // 100
                        self.memory[self.index_register + 1] = (value // 10) % 10
                        self.memory[self.index_register + 2] = value % 10
                        self.stats['memory_writes'] += 3
                elif kk == 0x55:  # LD [I], Vx (with memory quirk)
                    for i in range(x + 1):
                        if self.index_register + i < MEMORY_SIZE:
                            self.memory[self.index_register + i] = int(self.registers[i])
                    self.stats['memory_writes'] += x + 1
                    
                    # Memory quirk: increment I register
                    if self.quirks['memory']:
                        self.index_register = (self.index_register + x + 1) & 0xFFFF
                        
                elif kk == 0x65:  # LD Vx, [I] (with memory quirk)
                    for i in range(x + 1):
                        if self.index_register + i < MEMORY_SIZE:
                            self.registers[i] = int(self.memory[self.index_register + i])
                    self.stats['memory_reads'] += x + 1
                    
                    # Memory quirk: increment I register
                    if self.quirks['memory']:
                        self.index_register = (self.index_register + x + 1) & 0xFFFF
                else:
                    self.log_debug(f"ERROR: Unknown Fx{kk:02X} instruction at PC=0x{self.program_counter-2:03X}")
                    self.crashed = True
                    return False
            else:
                self.log_debug(f"ERROR: Unknown instruction 0x{instruction:04X} at PC=0x{self.program_counter-2:03X}")
                self.crashed = True
                return False
                
        except Exception as e:
            self.log_debug(f"EXCEPTION executing 0x{instruction:04X} at PC=0x{self.program_counter-2:03X}: {e}")
            self.crashed = True
            return False
            
        return True
    
    def _draw_sprite(self, x_reg: int, y_reg: int, height: int):
        """Draw a sprite at position (Vx, Vy) with given height"""
        # NOTE: Display wait quirk is complex and often causes more problems than it solves
        # For now, we'll implement it as a simple frame counter instead of real timing
        if self.quirks['display_wait']:
            # Simple frame-based limiting instead of real-time timing
            # This is much less aggressive than the timing-based approach
            self.stats['display_updates'] += 1
            if self.stats['display_updates'] % 2 == 0:  # Allow every other draw
                pass  # Continue with drawing
            else:
                # Still draw but track that we're limiting
                pass  # Don't actually skip - just track for stats
        
        vx = int(self.registers[x_reg]) % DISPLAY_WIDTH
        vy = int(self.registers[y_reg]) % DISPLAY_HEIGHT
        self.registers[0xF] = 0  # Clear collision flag
        
        for row in range(height):
            if vy + row >= DISPLAY_HEIGHT:
                break
                
            if self.index_register + row >= MEMORY_SIZE:
                break
                
            sprite_byte = int(self.memory[self.index_register + row])
            self.stats['memory_reads'] += 1
            
            for col in range(8):
                if vx + col >= DISPLAY_WIDTH:
                    break
                    
                if sprite_byte & (0x80 >> col):
                    pixel_y = vy + row
                    pixel_x = vx + col
                    
                    # Mark pixel as touched for statistics
                    if not self.pixel_touched[pixel_y, pixel_x]:
                        self.pixel_touched[pixel_y, pixel_x] = True
                        self.stats['unique_pixels_touched'] += 1
                    
                    # Check for collision
                    if self.display[pixel_y, pixel_x]:
                        self.registers[0xF] = 1
                        self.stats['sprite_collisions'] += 1
                        self.stats['pixels_erased'] += 1
                    else:
                        self.stats['pixels_drawn'] += 1
                    
                    # XOR the pixel
                    self.display[pixel_y, pixel_x] ^= 1
        
        self.stats['display_writes'] += 1
    
    def set_key(self, key: int, pressed: bool):
        """Set key state (0-F)"""
        if 0 <= key <= 0xF:
            self.keypad[key] = 1 if pressed else 0
            
            # Handle key waiting
            if self.waiting_for_key and pressed:
                self.registers[self.key_register] = key
                self.waiting_for_key = False
    
    def get_display(self) -> np.ndarray:
        """Get current display state as 2D array"""
        return self.display.copy()
    
    def get_display_as_image(self, scale: int = 8) -> np.ndarray:
        """Get display as a scaled image array suitable for display"""
        # Scale up the display
        scaled = np.repeat(np.repeat(self.display, scale, axis=0), scale, axis=1)
        # Convert to 0-255 range
        return scaled * 255
    
    def show_display(self, scale: int = 10, title: str = "CHIP-8 Display"):
        """Show display in a tkinter window with interactive input"""
        # Create tkinter window
        root = tk.Tk()
        root.title(title)
        root.resizable(False, False)
        
        # Flag to track if window is closing
        self._window_closing = False
        
        # Calculate window size
        canvas_width = DISPLAY_WIDTH * scale
        canvas_height = DISPLAY_HEIGHT * scale
        
        # Create canvas
        canvas = Canvas(root, width=canvas_width, height=canvas_height, bg='black')
        canvas.pack()
        
        # CHIP-8 keypad mapping to keyboard keys
        # Original CHIP-8 keypad:     Modern keyboard mapping:
        # 1 2 3 C                     1 2 3 4
        # 4 5 6 D          =>         Q W E R  
        # 7 8 9 E                     A S D F
        # A 0 B F                     Z X C V
        
        key_mapping = {
            '1': 0x1, '2': 0x2, '3': 0x3, '4': 0xC,
            'q': 0x4, 'w': 0x5, 'e': 0x6, 'r': 0xD,
            'a': 0x7, 's': 0x8, 'd': 0x9, 'f': 0xE,
            'z': 0xA, 'x': 0x0, 'c': 0xB, 'v': 0xF
        }
        
        # Track pressed keys for proper release handling
        pressed_keys = set()
        
        def key_press(event):
            if self._window_closing:
                return
            key = event.keysym.lower()
            if key in key_mapping:
                chip8_key = key_mapping[key]
                if chip8_key not in pressed_keys:
                    pressed_keys.add(chip8_key)
                    self.set_key(chip8_key, True)
                    print(f"Key pressed: {key} -> CHIP-8 key 0x{chip8_key:X}")
            elif key == 'escape':
                close_window()
        
        def key_release(event):
            if self._window_closing:
                return
            key = event.keysym.lower()
            if key in key_mapping:
                chip8_key = key_mapping[key]
                if chip8_key in pressed_keys:
                    pressed_keys.remove(chip8_key)
                    self.set_key(chip8_key, False)
                    print(f"Key released: {key} -> CHIP-8 key 0x{chip8_key:X}")
        
        def close_window():
            """Safely close the window and stop all callbacks"""
            self._window_closing = True
            root.quit()
            root.destroy()
        
        def update_display():
            """Update the display and continue emulation"""
            if self._window_closing:
                return
                
            try:
                # Clear canvas
                canvas.delete("all")
                
                # Draw pixels
                for y in range(DISPLAY_HEIGHT):
                    for x in range(DISPLAY_WIDTH):
                        if self.display[y, x]:
                            x1 = x * scale
                            y1 = y * scale
                            x2 = x1 + scale
                            y2 = y1 + scale
                            canvas.create_rectangle(x1, y1, x2, y2, fill='white', outline='white')
                
                # Continue emulation if not crashed
                if not self.crashed and not self.halt:
                    # Run a few cycles
                    for _ in range(10):  # Run 10 instructions per frame
                        if not self.step():
                            break
                    
                    # Update timers
                    if self.delay_timer > 0:
                        self.delay_timer -= 1
                    if self.sound_timer > 0:
                        self.sound_timer -= 1
                
                # Schedule next update (approximately 60 FPS) - only if window is still open
                if not self._window_closing:
                    root.after(16, update_display)
            except tk.TclError:
                # Window was destroyed, stop callbacks
                self._window_closing = True
        
        # Bind keyboard events
        root.bind('<KeyPress>', key_press)
        root.bind('<KeyRelease>', key_release)
        root.focus_set()  # Make sure window can receive key events
        
        # Handle window close button
        root.protocol("WM_DELETE_WINDOW", close_window)
        
        # Draw initial display
        for y in range(DISPLAY_HEIGHT):
            for x in range(DISPLAY_WIDTH):
                if self.display[y, x]:
                    x1 = x * scale
                    y1 = y * scale
                    x2 = x1 + scale
                    y2 = y1 + scale
                    canvas.create_rectangle(x1, y1, x2, y2, fill='white', outline='white')
        
        # Create instruction panel
        info_frame = tk.Frame(root)
        info_frame.pack(fill='x', padx=5, pady=5)
        
        # Keypad reference
        keypad_text = tk.Label(info_frame, 
            text="CHIP-8 Keypad Layout:\n" +
                 "1 2 3 4    →    1 2 3 C\n" +
                 "Q W E R    →    4 5 6 D\n" + 
                 "A S D F    →    7 8 9 E\n" +
                 "Z X C V    →    A 0 B F\n\n" +
                 "Press ESC to close",
            font=('Courier', 9), justify='left', bg='lightgray')
        keypad_text.pack(side='left')
        
        # Stats display
        stats_text = tk.Label(info_frame, text="", font=('Courier', 9), justify='right')
        stats_text.pack(side='right')
        
        def update_stats():
            """Update statistics display"""
            if self._window_closing:
                return
                
            try:
                stats_info = (f"PC: 0x{self.program_counter:03X}\n" +
                             f"I: 0x{self.index_register:03X}\n" +
                             f"Instructions: {self.stats['instructions_executed']}\n" +
                             f"Crashed: {self.crashed}")
                stats_text.config(text=stats_info)
                
                # Schedule next update - only if window is still open
                if not self._window_closing:
                    root.after(100, update_stats)  # Update every 100ms
            except tk.TclError:
                # Window was destroyed, stop callbacks
                self._window_closing = True
        
        # Start the update loops
        update_display()
        update_stats()
        
        # Show window
        try:
            root.mainloop()
        finally:
            # Ensure cleanup
            self._window_closing = True
    
    def get_stats(self) -> Dict[str, int]:
        """Get current instrumentation statistics"""
        return self.stats.copy()
    
    def print_stats(self):
        """Print current statistics"""
        print("CHIP-8 Emulator Statistics:")
        print("-" * 30)
        for key, value in self.stats.items():
            print(f"{key:25s}: {value}")

# Utility functions for testing
def load_rom_file(filename: str) -> bytes:
    """Load a ROM file"""
    with open(filename, 'rb') as f:
        return f.read()

def test_single_rom(rom_data: Union[str, bytes], cycles: int = 1000, display: bool = False):
    """Test a single ROM and optionally display results"""
    emulator = Chip8Emulator()
    emulator.load_rom(rom_data)
    emulator.run(max_cycles=cycles)
    
    print("Execution completed!")
    emulator.print_stats()
    
    if display:
        display_data = emulator.get_display()
        print("\nDisplay output:")
        for row in display_data:
            line = ''.join('██' if pixel else '  ' for pixel in row)
            print(line)
    
    return emulator

def test_rom_file(filename: str, cycles: int = 5000, scale: int = 8, show_display: bool = True, debug: bool = False, interactive: bool = False, quirks: dict = None):
    """Test a ROM file from disk"""
    try:
        # Create debug log file
        debug_file = None
        if debug:
            debug_file = f"chip8_debug_{os.path.basename(filename).replace('.ch8', '')}.log"
            # Clear previous log
            if os.path.exists(debug_file):
                os.remove(debug_file)
            print(f"Debug output will be written to: {debug_file}")
        
        print(f"Loading ROM: {filename}")
        emulator = Chip8Emulator(debug_file=debug_file, quirks=quirks)
        emulator.load_rom(filename)
        
        if quirks:
            print(f"Using quirks: {quirks}")
        
        if debug:
            emulator.log_debug(f"=== CHIP-8 Debug Log for {filename} ===")
            emulator.log_debug(f"ROM loaded, starting execution...")
        
        if interactive:
            print("Starting interactive mode...")
            print("Use keyboard for input (see window for keypad mapping)")
            print("Press ESC to exit")
            emulator.run_interactive(show_display=show_display, scale=scale, title=f"CHIP-8: {os.path.basename(filename)}")
        else:
            emulator.run(max_cycles=cycles)
            
            if debug:
                emulator.log_debug(f"=== Execution completed ===")
                emulator.log_debug(f"Final state:")
                emulator.log_debug(f"  Crashed: {emulator.crashed}")
                emulator.log_debug(f"  PC: 0x{emulator.program_counter:03X}")
                emulator.log_debug(f"  Instructions executed: {emulator.stats['instructions_executed']}")
                emulator.log_debug(f"  Display writes: {emulator.stats['display_writes']}")
                print(f"\nFull debug log saved to: {debug_file}")
            
            print("Execution completed!")
            emulator.print_stats()
            
            print(f"\nEmulator crashed: {emulator.crashed}")
            print(f"Program counter: 0x{emulator.program_counter:03X}")
            
            if show_display:
                print("\nShowing display in window...")
                emulator.show_display(scale=scale, title=f"CHIP-8: {filename}")
        
        return emulator
        
    except FileNotFoundError:
        print(f"ROM file not found: {filename}")
        return None
    except Exception as e:
        print(f"Error loading ROM: {e}")
        return None

def run_test_suite(test_dir: str = "../chip8-test-suite/bin"):
    """Run multiple test ROMs from a directory"""
    import os
    import glob
    
    # Try multiple possible locations for the test suite
    possible_paths = [
        test_dir,  # Default (relative to emulators/)
        "chip8-test-suite/bin",  # In current directory
        "../chip8-test-suite/bin",  # In parent directory
        "../../chip8-test-suite/bin",  # In grandparent directory
        os.path.join(os.path.dirname(__file__), "..", "chip8-test-suite", "bin")  # Relative to script
    ]
    
    test_path = None
    for path in possible_paths:
        if os.path.exists(path):
            test_path = path
            break
    
    if not test_path:
        print("Test directory not found. Tried:")
        for path in possible_paths:
            print(f"  - {os.path.abspath(path)}")
        print("\nDownload test suite with:")
        print("  git clone https://github.com/Timendus/chip8-test-suite.git")
        print("  (Put it in your babelscope root directory)")
        return
    
    print(f"Using test directory: {os.path.abspath(test_path)}")
    
    # Find all .ch8 files
    test_files = glob.glob(os.path.join(test_path, "*.ch8"))
    
    if not test_files:
        print(f"No .ch8 files found in {test_path}")
        return
    
    print(f"Found {len(test_files)} test ROMs:")
    for i, filename in enumerate(test_files):
        print(f"  {i+1}. {os.path.basename(filename)}")
    
    # Let user pick which test to run
    try:
        choice = input(f"\nEnter test number (1-{len(test_files)}), 'all', or 'interactive X' for interactive mode: ")
        
        if choice.lower() == 'all':
            for filename in test_files:
                print(f"\n{'='*50}")
                test_rom_file(filename, cycles=10000, scale=6, debug=True)
                input("Press Enter for next test...")
        elif choice.lower().startswith('interactive'):
            # Interactive mode for a specific test
            parts = choice.split()
            if len(parts) > 1:
                try:
                    test_num = int(parts[1]) - 1
                    if 0 <= test_num < len(test_files):
                        test_rom_file(test_files[test_num], interactive=True, scale=8)
                    else:
                        print("Invalid test number for interactive mode")
                except ValueError:
                    print("Invalid test number format")
            else:
                print("Please specify test number for interactive mode (e.g., 'interactive 1')")
        else:
            test_num = int(choice) - 1
            if 0 <= test_num < len(test_files):
                # Ask if user wants interactive mode
                interactive_choice = input("Run in interactive mode? (y/n): ").lower()
                interactive = interactive_choice.startswith('y')
                test_rom_file(test_files[test_num], cycles=10000, scale=8, debug=True, interactive=interactive)
            else:
                print("Invalid test number")
                
    except (ValueError, KeyboardInterrupt):
        print("Test cancelled")

def test_quirks_rom(filename: str = None):
    """Test the 5-quirks ROM with proper quirk settings"""
    if filename is None:
        # Try to find the 5-quirks ROM
        import os
        import glob
        possible_paths = [
            "../chip8-test-suite/bin/5-quirks.ch8",
            "chip8-test-suite/bin/5-quirks.ch8",
            "5-quirks.ch8"
        ]
        
        filename = None
        for path in possible_paths:
            if os.path.exists(path):
                filename = path
                break
        
        if filename is None:
            print("Could not find 5-quirks.ch8. Please specify the path.")
            return None
    
    print("Testing 5-quirks ROM with modern CHIP-8 quirks enabled...")
    
    # Configure quirks for modern CHIP-8 behavior
    modern_quirks = {
        'memory': True,      # Fx55/Fx65 increment I register  
        'display_wait': False, # Drawing waits for vblank - DISABLED (causes display issues)
        'jumping': True,     # Bnnn uses vX instead of v0
        'shifting': False,   # 8xy6/8xyE use vX (modern behavior)
        'logic': True,       # 8xy1/8xy2/8xy3 reset vF to 0
    }
    
    print("Quirks enabled:")
    for quirk, enabled in modern_quirks.items():
        print(f"  {quirk}: {'ON' if enabled else 'OFF'}")
    
    return test_rom_file(filename, cycles=20000, debug=True, quirks=modern_quirks)

# Example usage
if __name__ == "__main__":
    print("CHIP-8 Emulator for Babelscope")
    print("=" * 40)
    
    # Test quirks ROM first
    print("Testing CHIP-8 quirks...")
    test_quirks_rom()
    
    # Test with built-in program
    test_program = np.array([
        0xA2, 0x0A,  # LD I, 0x20A (point to sprite data at end of program)
        0x60, 0x0C,  # LD V0, 12 (x position)            
        0x61, 0x08,  # LD V1, 8  (y position)            
        0xD0, 0x15,  # DRW V0, V1, 5 (draw sprite)       
        0x12, 0x08,  # JP 0x208 (jump back to draw instruction - infinite loop)
        # Sprite data (simple pattern) - starts at 0x20A (index 10)
        0xF0, 0x90, 0x90, 0x90, 0xF0
    ], dtype=np.uint8)
    
    print("\nTesting built-in program...")
    emulator = test_single_rom(test_program, cycles=100, display=False)
    
    print(f"\nEmulator crashed: {emulator.crashed}")
    print(f"Program counter: 0x{emulator.program_counter:03X}")
    
    # Test interactive input with a simple input test program
    print("\n" + "="*50)
    print("Testing interactive input...")
    
    # Simple input test program
    input_test_program = np.array([
        0x00, 0xE0,  # CLS - clear screen
        0xF0, 0x0A,  # LD V0, K - wait for key press
        0xF0, 0x29,  # LD F, V0 - set I to font character for pressed key
        0x61, 0x10,  # LD V1, 16 - x position
        0x62, 0x10,  # LD V2, 16 - y position  
        0xD1, 0x25,  # DRW V1, V2, 5 - draw the character
        0x12, 0x02,  # JP 0x202 - jump back to wait for next key
    ], dtype=np.uint8)
    
    print("Creating input test emulator...")
    input_emulator = Chip8Emulator()
    input_emulator.load_rom(input_test_program)
    
    user_choice = input("Test interactive input? (y/n): ")
    if user_choice.lower().startswith('y'):
        print("Starting interactive input test...")
        print("Press any key (1,2,3,4,Q,W,E,R,A,S,D,F,Z,X,C,V) to see the corresponding hex digit!")
        print("Press ESC to exit")
        input_emulator.run_interactive(scale=10, title="CHIP-8 Input Test")
    
    print("\n" + "="*50)
    print("Now testing with ROM files...")
    
    # Try to run test suite
    run_test_suite()