![license: WTFPL-3](https://img.shields.io/badge/license-WTFPL_3-brightgreen)
![CUDA 12.x](https://img.shields.io/badge/cuda-12.x-blue)
![Not AI](https://img.shields.io/badge/Not-AI-informational)
![Made with CUDA](https://img.shields.io/badge/Made%20with-CUDA-green)

# Babelscope: Computational Archaeology with CUDA

This repository contains *Babelscope*, a system where we use GPUs to fill a virtual machine with random code, then execute it. Sometimes, the programs actually run. This is great, because this tool can emulate 200,000 independent instances of this VM at a time, find interesting stuff, then save these random program files for later inspection.

**A much better description of this project [is on my website](https://bbenchoff.github.io/pages/Babelscope.html).** If sharing on social media, I encourage you to post _that_.

## Table Of Contents

- [Introduction](#introduction)  
- [Experiment 1 – Random Games](#experiment-1-finding-random-games)  
- [Experiment 2: Discovering A Sorting Algorithm](#experiment-2-discovering-a-sorting-algorithm)


## Introduction

Following my [Finite Atari Project](https://bbenchoff.github.io/pages/FiniteAtari.html), I wondered if the technique could be improved to generate random ROMs for the Atari 2600 and find interesting things. I wanted more games that responded to input and the generation of novel algorithms. The only way to do this is to both generate programs on the GPU and emulate them in parallel. It's the halting problem; you don't know what a program is going to do until you run it.

This project eschews the Atari 2600 for the CHIP-8. It's a vastly simpler computer with a flatter memory model. Also you don't have to generate thousands of NTSC waveforms in a GPU.

For this project, I'm generating _billions_ of programs for the CHIP-8, and running them all in parallel. Instrumentation in the emulator allows me to select interesting ones on the fly (output to the display, responds to user input). Because the emulator is instrumented, I can also inspect memory while the emulator is running. With this, I'm able to find novel algorithms, like simple sorting algorithms.

This repo, or rather this README, is broken up into several sections, each detailing an experiment I conducted in massively parallel emulation of a virtual machine loaded with random data. Why? I'd suggest you [read the blog post that accompanies this project](https://bbenchoff.github.io/pages/Babelscope.html).

# Experiment 1: Finding Random Games

The inspiration for this project, [Finite Atari Project](https://bbenchoff.github.io/pages/FiniteAtari.html), generated billions of random ROMs for the Atari 2600 to find ones that would run. I found a bunch and one was even a proto-game, a program with visual output that responded to user input. Why not do the same with CHIP-8.

# Experiment 2: Discovering A Sorting Algorithm

The idea of this is simple. I generate billions of programs filled with random data, except for `[8 3 6 1 7 2 5 4]` at memory locations `0x300 to 0x307`. I inspect these programs while they're running. If I ever get `[1 2 3 4 5 6 7 8]` or `[8 7 6 5 4 3 2 1]`, I may have found a sorting algorithm. I might rediscover quicksort. I may find something else entirely. Who knows.

## Method

This experiment used a specially instrumented emulator, [sorting_emulator.py](emulators/sorting_emulator.py) which is a CUDA-based CHIP-8 emulator that provides real-time memory monitoring, and program capture with metadata. This script is used by a runner file, [sorting_search.py](sorting_search.py), that generates the program batches on the GPU, coordinates the parallel emulation effort, and handles discovery processing and analysis.

### How It Works

1. **Generate Random ROMs**: Creates completely random CHIP-8 programs (3584 bytes each)
2. **Setup Test Data**: Places the unsorted array `[8, 3, 6, 1, 7, 2, 5, 4]` at memory location 0x300-0x307
3. **Execute Programs**: Runs complete CHIP-8 emulation for each random program for 100,000 cycles
4. **Monitor for Sorting**: Checks periodically if the array becomes sorted to either:
   - `[1, 2, 3, 4, 5, 6, 7, 8]` (ascending)
   - `[8, 7, 6, 5, 4, 3, 2, 1]` (descending)
5. **Save Discoveries**: When sorting is detected, saves the ROM binary and metadata

### Performance

I bit the bullet and bought an RTX 5080 for this project:
- **ROM Generation**: ~70M ROMs/second on GPU
- **Emulation**: ~170K ROMs/second through complete CHIP-8 execution
- **Memory Usage**: ~7GB GPU memory for 200K parallel instances
- **GPU Utilization**: 90%+ sustained

### Requirements

- NVIDIA GPU with CUDA support
- Python 3.7+
- CuPy: `pip install cupy-cuda12x`
- NumPy: `pip install numpy`

### Quick Start

```bash
# Basic exploration
python sorting_search.py --batch-size 200000 --batches 100

# Continuous search (Ctrl+C to stop)
python sorting_search.py --batch-size 200000 --infinite

# High sensitivity detection (checks every 10 cycles instead of 100)
python sorting_search.py --batch-size 200000 --check-interval 10
```

### Command Line Options

```
--batch-size N          ROMs per batch (default: 50000)
--batches N              Number of batches (default: 10)
--infinite               Run infinite batches until interrupted
--cycles N               Execution cycles per ROM (default: 100000)
--check-interval N       Check for sorting every N cycles (default: 100)
--output-dir DIR         Output directory (default: babelscope_results)
--save-frequency N       Save session state every N batches (default: 10)
```

### Performance Tuning

**Batch Size**: Larger batches improve GPU utilization but use more memory. Try 100K-500K depending on your GPU.

**Check Interval**: Lower values catch more transient sorting but reduce performance:
- `--check-interval 1`: Perfect detection, ~30% slower
- `--check-interval 10`: Very good detection, ~10% slower  
- `--check-interval 100`: Default performance
- `--check-interval 1000`: Slightly faster, may miss brief sorting

**Cycles**: More cycles allow longer programs to complete sorting:
- `50000`: Fast, catches obvious sorting
- `100000`: Default balance
- `200000+`: Thorough, allows complex algorithms

### Architecture

The system uses a massively parallel CUDA kernel that implements complete CHIP-8 emulation across thousands of instances simultaneously. Each instance:

- Maintains full CHIP-8 state (memory, registers, stack, timers)
- Executes all 35 CHIP-8 instructions correctly
- Tracks memory access patterns
- Monitors for the specific sorting condition
- Terminates early when sorting is detected

This approach allows exploration of the random program space at unprecedented scale, making computational archaeology practical for the first time.

## Results

THE RESULTS GO HERE WHEN I FIND THEM

## Discussion

I'm of two minds about the fact that I found a sorting algorithm in random data. Firstly, _of course I would_. There are billions of ways to write an algorithm that would sort the data between `0x300` and `0x307`. After emulating billions of ROMs, _something_ interesting was bound to happen. On the other hand, this is _very weird_. This wasn't created, because it's just a pile of random data that happened to do something. It was just there in the huge computational space of all possible CHIP-8 programs. This isn't computer science, it's more like computer archaeology. Or astronomy.

# License

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