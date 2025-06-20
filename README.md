# Babelscope

**Computational Archaeology for the CHIP-8**

Babelscope generates billions of random CHIP-8 programs and runs them in parallel on GPU hardware to find accidental algorithms, graphics demos, and proto-games hidden in the vast space of random machine code.

## What Does This Do?

Instead of writing programs, Babelscope **discovers** them by:

1. **Generating pure random data** (3584 bytes each) 
2. **Testing millions simultaneously** on GPU using a massively parallel CHIP-8 emulator
3. **Finding the ones that work** - programs that don't crash and produce visual output
4. **Saving the interesting ones** with screenshots for analysis

Think of it as pointing a telescope at the infinite library of possible programs and cataloging the strange objects you find.

## Results

Out of millions of random byte sequences, Babelscope finds programs that:
- Draw structured graphics patterns
- Create animations and visual effects  
- Run stable loops without crashing
- Occasionally respond to input (accidental proto-games!)

These aren't hand-coded - they're random data that accidentally works as CHIP-8 programs.

## Quick Start

**Generate and test random programs:**
```bash
python test_random_roms.py --continuous --batch-size 15000 --save-interesting
```

**View discovered programs:**
```bash
python view_interesting_roms.py
```

## Requirements

- Python 3.8+
- CuPy (for GPU acceleration)
- PIL (for screenshots)
- CUDA-capable GPU

## How It Works

### The Generator
`generators/random_chip8_generator.py` - Creates pure random data on GPU
- No heuristics, no intelligence
- Just 3584 bytes of random noise per "ROM"
- Generated in parallel across thousands of GPU threads

### The Emulator  
`emulators/mega_kernel_chip8.py` - Massively parallel CHIP-8 emulator
- Runs 100,000+ instances simultaneously on GPU
- Each thread emulates a complete CHIP-8 system
- 350+ million instructions per second throughput

### The Explorer
`test_random_roms.py` - Finds programs that accidentally work
- Tests random programs for crashes vs. completion
- Identifies visual output and structured patterns
- Saves interesting discoveries with screenshots

### The Viewer
`view_interesting_roms.py` - Interactive exploration of discoveries
- Loads found programs in CHIP-8 emulator windows
- Test for input responsiveness
- See what random data accidentally created

## Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Random Data   │───▶│  Mega-Kernel     │───▶│   Interesting   │
│   Generator     │    │  CHIP-8 Emulator │    │   ROMs + PNGs   │ 
│                 │    │                  │    │                 │
│ • Pure randomness│    │ • 100K+ parallel│    │ • Working programs│
│ • GPU-generated │    │ • 350M+ inst/sec│    │ • Visual output  │
│ • No filtering  │    │ • Crash detection│    │ • Screenshots    │
└─────────────────┘    └──────────────────┘    └─────────────────┘
```

## Performance

- **Generation**: 100,000+ random ROMs per second
- **Testing**: 350+ million CHIP-8 instructions per second  
- **Discovery rate**: ~0.01-0.1% of random data produces "interesting" programs
- **Scalability**: Nearly linear scaling with GPU cores until memory saturation

## Files

- `generators/random_chip8_generator.py` - Pure random ROM generation
- `emulators/mega_kernel_chip8.py` - Massively parallel CHIP-8 emulator
- `emulators/chip8.py` - Single-instance emulator for interactive testing
- `test_random_roms.py` - Main discovery pipeline
- `view_interesting_roms.py` - Interactive ROM viewer

## License

```
           DO WHAT THE FUCK YOU WANT TO PUBLIC LICENSE
                    MODIFIED FOR NERDS 
                   Version 3, April 2025

Everyone is permitted to copy and distribute verbatim or modified
copies of this license document, and changing it is allowed as long
as the name is changed.
 
           DO WHAT THE FUCK YOU WANT TO PUBLIC LICENSE
  TERMS AND CONDITIONS FOR COPYING, DISTRIBUTION AND MODIFICATION

 0. You just DO WHAT THE FUCK YOU WANT TO.

 1. Anyone who complains about this license is a nerd.
```
