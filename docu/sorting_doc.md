# Async Streaming Partial Sorting Babelscope

A high-performance GPU-accelerated system for discovering partial sorting algorithms in random CHIP-8 programs using async streaming and pipeline optimization.

## Overview

The Partial Sorting Babelscope searches for emergent sorting behavior in randomly generated CHIP-8 programs by detecting **partial consecutive sorting** in registers V0-V7. Unlike traditional approaches that only detect complete sorting, this system captures incremental progress toward sorting, dramatically increasing discovery probability.

### Key Enhancements

- **Partial Sorting Detection**: Finds 3+ consecutive sorted elements (e.g., `[1,2,3]`, `[8,7,6]`, `[2,3,4,5,6]`)
- **Async GPU Pipeline**: 3 CUDA streams + background processing for 80-90% GPU utilization
- **Streaming Architecture**: Overlapped ROM generation, kernel execution, and memory transfer
- **Performance Optimized**: 2-3x throughput improvement over synchronous approaches

## System Architecture

### GPU Pipeline (3 CUDA Streams)
```
Stream 1: ROM Generation     ──┐
Stream 2: Kernel Execution   ──┼─► Overlapped Processing
Stream 3: Memory Transfer    ──┘
Background: File I/O + Discovery Analysis
```

### Detection Method
- **Target**: Registers V0-V7 with test pattern `[8, 3, 6, 1, 7, 2, 5, 4]`
- **Detection**: 3+ consecutive sorted elements in either direction
- **Examples**: 
  - Ascending: `[1,2,3]`, `[1,2,3,4]`, `[2,3,4,5,6]`
  - Descending: `[8,7,6]`, `[7,6,5,4]`, `[6,5,4,3,2]`
- **Saving**: Only sequences of 6+ elements are saved to disk

## Installation

### Requirements
- Python 3.8+
- NVIDIA GPU with CUDA support
- CuPy for GPU acceleration
- NumPy for CPU operations

### Setup
```bash
# Install CuPy (adjust for your CUDA version)
pip install cupy-cuda12x  # For CUDA 12.x
# or
pip install cupy-cuda11x  # For CUDA 11.x

# Install other dependencies
pip install numpy
```

### File Structure
```
Babelscope/
├── sorting_search.py          # Main runner script
├── emulators/
│   └── sorting_emulator.py    # Core CHIP-8 emulator with sorting detection
└── output/
    └── async_partial_sorting/
        └── session_YYYYMMDD_HHMMSS/
            ├── discovered_roms/    # Found sorting ROMs
            └── logs/              # Session data and stats
```

## Usage

### Quick Test
```bash
# Validate system functionality
python sorting_search.py --test-mode
```

### Basic Usage
```bash
# Run 10 batches of 50K ROMs each
python sorting_search.py --batch-size 50000 --batches 10

# Run infinite exploration (Ctrl+C to stop)
python sorting_search.py --batch-size 100000 --infinite

# High-sensitivity detection (slower but more thorough)
python sorting_search.py --batch-size 50000 --check-interval 50
```

### Advanced Options
```bash
python sorting_search.py \
    --batch-size 200000 \
    --batches 100 \
    --cycles 200000 \
    --check-interval 100 \
    --output-dir custom_output \
    --save-frequency 5
```

## Command Line Options

| Option | Default | Description |
|--------|---------|-------------|
| `--batch-size` | 50000 | ROMs per batch |
| `--batches` | 10 | Number of batches to run |
| `--infinite` | False | Run until interrupted |
| `--cycles` | 100000 | Execution cycles per ROM |
| `--check-interval` | 100 | Check for sorting every N cycles |
| `--output-dir` | `output/async_partial_sorting` | Output directory |
| `--save-frequency` | 10 | Save state every N batches |
| `--test-mode` | False | Quick validation test |
| `--skip-validation` | False | Skip GPU validation |

## Performance Characteristics

### Expected Throughput
- **Target**: 80K-120K ROMs/sec (vs 40K baseline)
- **GPU Utilization**: 80-90% (vs 30-40% peaky)
- **Discovery Rate**: 1 in 10K-100K ROMs (varies by parameters)

### Memory Requirements
- **Minimum**: 4GB GPU memory
- **Recommended**: 8GB+ for large batch sizes
- **Scaling**: ~80MB per 100K ROMs

### Performance Tuning
- **Batch Size**: Larger = better GPU utilization, more memory
- **Check Interval**: Lower = more sensitive, slower execution
- **Cycles**: Higher = more thorough search per ROM

## Output Files

### Discovered ROMs
Each discovery generates:
- `*.ch8`: Binary ROM file
- `*.json`: Metadata with sorting details

### Filename Format
```
LONGPARTIAL_B{batch}D{discovery}_{range}_L{length}_{direction}_C{cycle}_{hash}.ch8
```
Example: `LONGPARTIAL_B0042D01_V0-V5_L6_ASC_C45231_a7b3c4d2.ch8`

### Session Data
- `async_partial_sorting_session_state.json`: Detailed session state
- `async_partial_sorting_summary.txt`: Human-readable summary
- `babelscope_sorting_stats.json`: Global statistics across all sessions

## Discovery Examples

### Sample Output
```
BATCH 42
Loading 100,000 random ROMs from GPU array...
Processing discovered ROMs...
   LONGPARTIAL_B0042D01_V0-V5_L6_ASC_C45231_a7b3c4d2: V0-V5 (6 elements, ascending)
      Sequence: [1, 2, 3, 4, 5, 6] @ cycle 45,231
   LONGPARTIAL_B0042D02_V2-V7_L6_DESC_C67543_f8e9d0c1: V2-V7 (6 elements, descending)
      Sequence: [8, 7, 6, 5, 4, 3] @ cycle 67,543

Batch 42 summary:
   ROMs tested: 100,000
   Partial checks: 1,000,000
   Total discoveries: 8
   Long sequences saved: 2 (length 6+)
```

### Discovery Metadata
```json
{
  "discovery_info": {
    "batch": 42,
    "discovery_number": 1,
    "instance_id": 23847,
    "sort_cycle": 45231,
    "discovery_type": "long_partial_consecutive_sorting"
  },
  "partial_sorting": {
    "length": 6,
    "start_position": 0,
    "end_position": 5,
    "direction": "ascending",
    "sequence": [1, 2, 3, 4, 5, 6],
    "sequence_range": "V0-V5"
  },
  "registers": {
    "initial": [8, 3, 6, 1, 7, 2, 5, 4],
    "final": [1, 2, 3, 4, 5, 6, 5, 4]
  }
}
```

## Technical Details

### CHIP-8 Emulator
- **Complete Implementation**: All 35 CHIP-8 instructions
- **Memory**: 4KB per instance
- **Registers**: 16 general-purpose (V0-VF)
- **Display**: 64x32 monochrome
- **Timers**: Delay and sound timers

### Sorting Detection Algorithm
1. **Setup**: Load test pattern `[8, 3, 6, 1, 7, 2, 5, 4]` into V0-V7
2. **Execution**: Run random CHIP-8 program
3. **Detection**: Check for consecutive sorted sequences every N cycles
4. **Validation**: Verify minimum length requirements
5. **Recording**: Save discoveries with full metadata

### GPU Optimizations
- **Register Caching**: Cache V0-V7 values for faster sorting checks
- **Early Termination**: Stop searching once sorting is found
- **Reduced Branching**: Optimized CUDA kernel structure
- **Memory Coalescing**: Efficient GPU memory access patterns

## Troubleshooting

### Common Issues

**GPU Memory Errors**
```bash
# Reduce batch size
python sorting_search.py --batch-size 25000
```

**Low Performance**
```bash
# Check GPU utilization
nvidia-smi

# Increase batch size if GPU underutilized
python sorting_search.py --batch-size 200000
```

**No Discoveries**
```bash
# Increase sensitivity (slower but more thorough)
python sorting_search.py --check-interval 5 --cycles 200000
```

### Validation Commands
```bash
# Test GPU and environment
python sorting_search.py --test-mode

# Check CUDA streams
python -c "import cupy as cp; s = cp.cuda.Stream(); print('CUDA streams OK')"

# Verify memory
python -c "import cupy as cp; print(f'GPU memory: {cp.cuda.Device().mem_info[1]//1024**3}GB')"
```

## Performance Analysis

### Benchmarking
The system tracks comprehensive performance metrics:
- ROMs/second processing rate
- GPU utilization percentage
- Discovery rates and patterns
- Memory usage and efficiency

### Expected Results
- **Discovery Rate**: Highly variable, typically 1 in 10K-100K ROMs
- **Processing Speed**: 80K-120K ROMs/sec on modern GPUs
- **Memory Efficiency**: ~800 bytes per ROM instance
- **Pipeline Efficiency**: 80-90% GPU utilization

## Contributing

### Adding New Detection Methods
1. Extend the CUDA kernel in `sorting_emulator.py`
2. Add detection logic in the main execution loop
3. Update result processing and metadata
4. Test with the validation framework

### Performance Improvements
- Optimize memory access patterns
- Reduce CUDA kernel divergence
- Improve async pipeline coordination
- Add additional CUDA streams for specific operations

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

## Citation

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
