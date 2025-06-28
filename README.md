[![DOI](https://zenodo.org/badge/1000519368.svg)](https://doi.org/10.5281/zenodo.15741812)
![Not AI](https://img.shields.io/badge/Not-AI-informational)
![license: WTFPL-3](https://img.shields.io/badge/license-WTFPL_3-brightgreen)
![CUDA 12.x](https://img.shields.io/badge/cuda-12.x-blue)
![Made with CUDA](https://img.shields.io/badge/Made%20with-CUDA-green)

# Babelscope: Finding algorithms in random data with CUDA

This repository contains *Babelscope*, a system where we use GPUs to fill a virtual machine with random code, then execute it. Sometimes, the programs actually run. This is great, because this tool can emulate 500,000 independent instances of this VM at a time, find interesting stuff, then save these random program files for later inspection.

This is neither an evolutionary algorithm, nor is it any other type of machine learning or AI. This is the computer science equivalent of the [Miller-Urey experiment](https://en.wikipedia.org/wiki/Miller%E2%80%93Urey_experiment): Instead of finding amino acids in the primordial soup, I found tiny algorithms bubbling up in random data.

### **A much better description of this project [is on my website](https://bbenchoff.github.io/pages/Babelscope.html).** If sharing on social media, I encourage you to post _that_.

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

The inspiration for this project, [Finite Atari Project](https://bbenchoff.github.io/pages/FiniteAtari.html), generated billions of random ROMs for the Atari 2600 to find ones that would run. I found a bunch and one was even a proto-game, a program with visual output that responded to user input. Why not do the same with CHIP-8?

![screencaps of emulators running after 100,000 steps](https://bbenchoff.github.io/images/Bablescope/rom_mosaic.png)

These are about what you would expect. There's some interesting visual output, but it's _just random data_. There's nothing interesting going on here. I already proved this could happen with the Finite Atari Machine, and the 'protogames' from that looked cooler anyway.

Consider this a verification of the technique, but an absolute failure of doing anything cool. An actual in-browser emulation of some of these programs is available [on my website](https://bbenchoff.github.io/pages/Babelscope.html).

# Experiment 2: Discovering A Sorting Algorithm

The idea of this is simple. I generate billions of programs filled with random data, except for `[8 3 6 1 7 2 5 4]` stored in the CHIP-8 registers. V0 (register zero) stores a value of 8, V1 (register 1) stores a value of 3... all the way up to V7 (register 7), which stores a value of 4. Now fill the program with random data, and emulate it. Every few cycles, check the values of the registers to see if the entire thing is sorted (either `[1 2 3 4 5 6 7 8]` or `[8 7 6 5 4 3 2 1]`), or any sub-strings are sorted `[8 7 6 5 1 4 3 2]` -- the first four items are sorted in reverse order.

## Method

This experiment used a specially instrumented emulator, [sorting_emulator.py](emulators/sorting_emulator.py) which is a CUDA-based CHIP-8 emulator that provides real-time memory monitoring, and program capture with metadata. This script is used by a runner file, [sorting_search.py](sorting_search.py), that generates the program batches on the GPU, coordinates the parallel emulation effort, and handles discovery processing and analysis.

### How It Works

1. **Generate Random ROMs**: Creates completely random CHIP-8 programs (3584 bytes each)
2. **Setup Test Data**: Places the unsorted array `[8, 3, 6, 1, 7, 2, 5, 4]` in registers V0 to V7.
3. **Execute Programs**: Runs complete CHIP-8 emulation for each random program for 100,000 cycles
4. **Monitor for Sorting**: Checks periodically if the registers becomes sorted to either:
   - `[1, 2, 3, 4, 5, 6, 7, 8]` (ascending)
   - `[8, 7, 6, 5, 4, 3, 2, 1]` (descending)
   - any sub-string sort, for example `[1 2 3 4 8 5 6 7]`, which is the first four elements sorted in ascending order
5. **Save Discoveries**: When sorting is detected, saves the ROM binary and metadata. I'm only saving sorts with a substring length > 6, for ease of processing.

My first test in this experiment populated the entire program with random data, except for a single memory segment from `0x300` to `0x307`, which was filled with `[8, 3, 6, 1, 7, 2, 5, 4]`. Running this for ~11 hours generated **3,903,200,000 random programs** at a rate of **100,761 programs/second**. Nothing was found. It was a negative result, although that doesn't really mean anything because I barely scratched the surface of the space of possible programs.

The next test fixed the obvious problem. Instead of testing one memory location, I put data where the CHIP-8 opcods could actually read it: in the registers. The CHIP-8 has 16 general purpose 8-bit register, named _Vx_ where _x_ is a digit from 0 to F. The _VF_ register is reserved, so I put `[8, 3, 6, 1, 7, 2, 5, 4]` in registers _V0_ through _V7_.

A runner script creates hundreds of thousands of these programs, full of random data, and puts values in those registers. These programs are then loaded into the emulators in the GPU, run in parallel. The registers are checked every 2-4 instructions (it's a very computationally expensive process), and any time the registers, or a subset of the registers, are sorted either ascending or descending, that program is sent back to the runner script. Results are cataloged and saved.

Do this for a few days, and you'll have a dozen or so perfectly sorted registers, and a couple of near-misses, with six or seven registers sorted.

## Results

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

---

## Cite as

**APA**

Benchoff, B. (2025). *Babelscope: Finding algorithms in random data with CUDA* (Version v0.1.0) [Computer software]. GitHub. https://github.com/bbenchoff/Babelscope

**BibTeX**

@software{benchoff2025babelscope,
  author  = {Benchoff, Brian},
  title   = {{Babelscope: Finding Algorithms in Random Data with CUDA}},
  year    = {2025},
  version = {v0.1.0},
  url     = {https://github.com/bbenchoff/Babelscope},
  note    = {GPU-scale computational archaeology},
}
