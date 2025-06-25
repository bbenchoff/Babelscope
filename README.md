# Babelscope: Computational Archaeology with CUDA

## Sorting Algorithm Discovery

Babelscope explores the vast space of random machine code to discover emergent computational behaviors. The sorting algorithm discovery tool generates millions of random CHIP-8 programs and searches for those that accidentally implement sorting algorithms.

### How It Works

1. **Generate Random ROMs**: Creates completely random CHIP-8 programs (3584 bytes each)
2. **Setup Test Data**: Places the unsorted array `[8, 3, 6, 1, 7, 2, 5, 4]` at memory location 0x300-0x307
3. **Execute Programs**: Runs complete CHIP-8 emulation for each random program
4. **Monitor for Sorting**: Checks periodically if the array becomes sorted to either:
   - `[1, 2, 3, 4, 5, 6, 7, 8]` (ascending)
   - `[8, 7, 6, 5, 4, 3, 2, 1]` (descending)
5. **Save Discoveries**: When sorting is detected, saves the ROM binary and metadata

This is computational archaeology - excavating working algorithms from the fossil record of random bit sequences.

### Performance

On an RTX 5080:
- **ROM Generation**: ~70M ROMs/second on GPU
- **Emulation**: ~167K ROMs/second through complete CHIP-8 execution
- **Memory Usage**: ~7GB GPU memory for 200K parallel instances
- **GPU Utilization**: 90%+ sustained

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

### Output Structure

```
babelscope_results/
└── session_YYYYMMDD_HHMMSS/
    ├── discovered_roms/
    │   ├── FOUND_B0001D01_C1234_abcd1234.ch8    # Actual ROM binary
    │   ├── FOUND_B0001D01_C1234_abcd1234.json   # Discovery metadata
    │   └── ...
    └── logs/
        ├── session_state.json                    # Complete session data
        └── summary.txt                          # Human-readable summary
```

### Discovery Metadata

Each discovered ROM includes:
- **ROM Binary**: The actual random machine code that achieved sorting
- **Discovery Info**: When and how it was found
- **Arrays**: Initial `[8,3,6,1,7,2,5,4]` and final sorted state
- **Memory Access**: Number of reads/writes to the test array
- **Cycle Count**: When during execution sorting was achieved

### Requirements

- NVIDIA GPU with CUDA support
- Python 3.7+
- CuPy: `pip install cupy-cuda12x`
- NumPy: `pip install numpy`

### Expected Discovery Rate

Finding algorithms that transform `[8,3,6,1,7,2,5,4]` into exactly `[1,2,3,4,5,6,7,8]` or `[8,7,6,5,4,3,2,1]` is extremely rare. Expect discovery rates around:

- **1 in 50-100 million ROMs** for legitimate sorting algorithms
- **Processing time**: Days to weeks to find first discovery
- **Storage**: Each discovery is ~4KB ROM + metadata

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