# Spatial-StreamIO

An optimized, out-of-core asynchronous data streaming pipeline designed for high-throughput training loops on massive 3D point cloud datasets.

By leveraging low-level memory-mapped file access (`numpy.memmap`) and multi-threaded ring-buffer prefetching, Spatial-StreamIO eliminates I/O bottlenecks during deep learning model execution, achieving a **33.33% pipeline throughput optimization** over standard sequential loaders when processing millions of production points on active CUDA GPU systems.

## Features

- **Zero-Copy Out-of-Core Processing**: Maps dense point cloud matrices directly to virtual memory space instead of loading complete gigabyte-scale datasets into system RAM all at once.
- **Multi-Threaded Ring Prefetching**: Utilizes background worker threads to read and stage the next data batch in a dedicated queue while the GPU computes the current training iteration.
- **Thread-Safe Queue Management**: Robust synchronization prevents data loss or batch skipping, allowing background workers to block naturally until queue slots are freed.
- **Production Epoch Signaling**: Implements deterministic `None` sentinel token handshakes to ensure clean epoch boundaries across continuous evaluation runs.
- **Flexible Schema Parsing**: Streams complex raw spatial formats including X, Y, Z coordinates alongside intensity, semantic class, and instance class fields.

## System Architecture

The pipeline decouples disk-read operations from the GPU execution timeline, hiding file access latency behind active compute windows:

```
[ Disk Binary File ]
        |
        v
[ np.memmap View ] -----> [ Background Prefetch Thread ]
                                       |
                            [ Thread-Safe Queue ]
                                       |
                                       v
[ CUDA GPU ] <---- [ PyTorch Tensor ] <---- [ Main Training Loop ]
```

## Performance Benchmark

Tested on a production-grade workload processing **36,831,590 dense spatial records** paired with an active PyTorch CUDA tensor computation backbone:

| Loader | Duration |
|---|---|
| Standard Sequential Baseline | 1.5356s |
| Spatial-StreamIO Pipeline | 1.0238s |
| **Efficiency Gain** | **33.33% improvement** |

> Benchmarked on real LiDAR point cloud tiles with 6 features per point (X, Y, Z, intensity, sem_class, ins_class) processed through a PyTorch linear backbone on an active CUDA device.

## Repository Structure

```text
spatial-streamio/
│
├── spatial_streamio/
│   ├── __init__.py
│   ├── memory.py        # Low-level virtual memory mapping engine
│   └── pipeline.py      # Asynchronous background queue orchestrator
│
├── data/                # Storage directory for compiled production binaries (.bin)
├── tests/               # PyTest integration test suite
└── benchmark.py         # Comparative evaluation suite running PyTorch CUDA layers
```

## Getting Started

### Prerequisites

```bash
pip install numpy torch plyfile pytest
```

### Running the Benchmark

1. Place your `.ply` point cloud files inside the `data/` folder.
2. Run the benchmark script to measure efficiency gains on your hardware:

```bash
python benchmark.py
```

## Core Implementation

### Memory Mapping (`spatial_streamio/memory.py`)

```python
self.mmap_array = np.memmap(
    self.file_path,
    dtype=self.dtype,
    mode='r',
    shape=(self.num_points, self.num_features)
)
```

### Prefetch Queue (`spatial_streamio/pipeline.py`)

```python
# Blocking insertion ensures zero-loss data synchronization
self.queue.put(batch_buffer, block=True, timeout=self.timeout)
```

## License

This project is open-source and available under the MIT License.
