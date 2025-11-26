# TorchComms: Complete Guide

A comprehensive guide to understanding and using TorchComms - Meta's next-generation communications API for PyTorch distributed training and inference.

---

## Table of Contents

1. [What is TorchComms?](#what-is-torchcomms)
2. [Fundamentals and Architecture](#fundamentals-and-architecture)
3. [Motivation](#motivation)
4. [Key Advantages](#key-advantages)
5. [Who Should Use TorchComms?](#who-should-use-torchcomms)
6. [Main Concepts](#main-concepts)
7. [Getting Started](#getting-started)
8. [Usage Examples](#usage-examples)
9. [Configuration](#configuration)
10. [Performance Characteristics](#performance-characteristics)

---

## What is TorchComms?

**TorchComms** is an experimental communications API for PyTorch that provides a unified, high-level interface for distributed training and inference across multiple GPUs and devices. It abstracts the complexity of underlying communication backends while supporting both synchronous and asynchronous operations across multiple GPU types (NVIDIA, AMD) and network topologies.

### Key Characteristics

- **Language**: Written in C++ with Python bindings (using pybind11)
- **Purpose**: Next-generation communications library for PyTorch
- **Installation**: Available as a Python package (pip or from source)
- **Backends**: Multiple pluggable backends (NCCL, RCCL, Gloo)
- **Scale**: ~3,725 lines of core C++ code in the main module

---

## Fundamentals and Architecture

### Architecture Overview

TorchComms follows a layered architecture with clear separation of concerns:

```
┌───────────────────────────────────────────────┐
│   Python API (torchcomms._comms module)       │ ← TorchCommPy.cpp (pybind11)
├───────────────────────────────────────────────┤
│   TorchComm (Unified Interface)               │ ← TorchComm.hpp/cpp
├───────────────────────────────────────────────┤
│   TorchCommFactory (Dynamic Backend Loading)  │ ← TorchCommFactory.cpp
├───────────────────────────────────────────────┤
│   Backend Implementations (Pluggable)         │
│   ├─ NCCL (Standard NVIDIA Collective Comms)  │
│   ├─ NCCLX (Extended NCCL with optimizations) │
│   ├─ RCCL (AMD ROCm Collective Communications)│
│   ├─ RCCLX (Extended RCCL with optimizations) │
│   └─ Gloo (CPU fallback backend)              │
├───────────────────────────────────────────────┤
│   Transport Layer                             │
│   ├─ RdmaTransport (RDMA over RoCE)           │
│   ├─ IBVerbX (InfiniBand abstraction)         │
│   └─ CTRAN (Modular collective comms)         │
└───────────────────────────────────────────────┘
```

### Design Patterns

#### 1. Factory Pattern (`TorchCommFactory`)
- **Purpose**: Dynamically loads backend implementations
- **Mechanism**: Uses `dlopen`/`dlsym` for runtime loading
- **Features**:
  - Maintains a registry of loaded backends
  - Supports ABI versioning for compatibility
  - Enables hot-swapping of backends without recompilation

#### 2. Backend Interface (`TorchCommBackend`)
- **Purpose**: Abstract base class for all backends
- **Benefits**:
  - Defines common API for all communication operations
  - Allows seamless backend switching at runtime
  - Enables custom backend implementations

#### 3. Wrapper Pattern (`TorchComm`)
- **Purpose**: Wraps backend implementation with a unified interface
- **Features**:
  - Delegates all operations to backend implementations
  - Provides consistent API across all backends
  - Handles common setup and teardown logic

#### 4. Asynchronous Work Objects (`TorchWork`)
- **Purpose**: Handles async operation completion
- **Features**:
  - Intrusive pointer-based reference counting
  - `wait()` and `is_completed()` methods for synchronization
  - CUDA stream integration

### Project Structure

```
torchcomms/
├── comms/torchcomms/           # Main library
│   ├── TorchComm.hpp/cpp       # Core communicator
│   ├── TorchCommBackend.hpp    # Backend interface
│   ├── TorchCommFactory.cpp    # Backend factory
│   ├── TorchCommPy.cpp         # Python bindings
│   ├── nccl/                   # NCCL backend
│   ├── ncclx/                  # Extended NCCL
│   ├── rccl/                   # AMD RCCL backend
│   ├── rcclx/                  # Extended RCCL
│   ├── gloo/                   # CPU fallback
│   ├── transport/              # RDMA transport
│   ├── examples/               # Usage examples
│   └── tests/                  # Test suite
├── comms/ctran/                # Generic collective transport
│   └── ibverbx/                # InfiniBand wrapper
└── comms/ncclx/v2_27/         # Extended NCCL source
```

---

## Motivation

### Background: The CTRAN Story

From the CTRAN README:
> "CTRAN emerged as a solution to challenges in NCCL, providing a modular, self-contained architecture for collective communications across different GPU types (NVIDIA, AMD) and network topologies."

### Primary Motivations

1. **Unified Interface**
   - Single API across multiple backends
   - Eliminates learning curve for backend-specific APIs
   - Reduces code duplication in distributed training

2. **Multi-GPU Type Support**
   - NVIDIA GPUs (via NCCL)
   - AMD GPUs (via RCCL)
   - CPU fallback (via Gloo)
   - All in one framework

3. **Hardware Flexibility**
   - InfiniBand support
   - RDMA over Converged Ethernet (RoCE)
   - Standard Ethernet
   - Optimized for each network topology

4. **Performance Optimization**
   - Extended backends (NCCLX, RCCLX) with additional optimizations
   - Support for advanced features beyond standard libraries
   - Zero-copy GPU-to-GPU transfers

5. **Distributed Training Scale**
   - Efficient scaling across multiple nodes and devices
   - Support for large-scale model training (LLMs, vision models)
   - Advanced collective communication patterns

6. **Research & Experimentation**
   - Experimental nature allows innovation
   - Test new collective communication algorithms
   - Benchmark different backend implementations

---

## Key Advantages

### 1. Multi-Backend Support

TorchComms supports multiple communication backends out of the box:

| Backend | GPU Type | Description |
|---------|----------|-------------|
| **NCCL** | NVIDIA | Standard NVIDIA collective communications |
| **NCCLX** | NVIDIA | Extended NCCL with profiling, tuning, networking extensions |
| **RCCL** | AMD | ROCm collective communications for MI series GPUs |
| **RCCLX** | AMD | Extended RCCL with optimizations |
| **Gloo** | CPU | CPU-based fallback for testing and small-scale work |

### 2. Asynchronous Operations

Enable computation-communication overlap for better performance:

```python
# Start communication
work = torchcomm.all_reduce(tensor, ReduceOp.SUM, async_op=True)

# Do other computation while communication happens
compute_forward_pass()

# Wait for communication to complete
work.wait()
```

**Benefits**:
- Non-blocking communication
- Better GPU utilization
- Reduced training time
- CUDA graph support for capturing repeated patterns

### 3. Flexible Communication Patterns

**Point-to-Point Operations**:
- `send()` / `recv()` - Basic message passing
- `batch_send()` / `batch_recv()` - Batched operations

**Collective Operations**:
- `broadcast()` - One-to-all communication
- `reduce()` - All-to-one with reduction
- `all_reduce()` - All-to-all with reduction (most common)
- `all_gather()` - Gather data from all ranks
- `reduce_scatter()` - Reduce and distribute
- `all_to_all()` - Complete exchange
- `scatter()` - One-to-all distribution
- `gather()` - All-to-one collection

**Advanced Patterns**:
- Subcommunicators via `split()`
- Object collectives (`objcol.all_gather_object()`)
- Window-based one-sided operations (RMA)

### 4. Advanced Features

- **CUDA Graph Support**: Capture communication in graphs for optimized replay
- **Backend Hints**: Fine-grained control over backend behavior
- **Timeout Configuration**: Per-operation or global timeout settings
- **Automatic Discovery**: Integration with Torchrun for multi-node setup
- **Custom Stores**: TCPStore, FileStore, or custom coordination

### 5. Developer Experience

- **Pythonic API**: Clean, intuitive method names
- **Type Hints**: Full typing support via `.pyi` stubs
- **Comprehensive Error Handling**: Clear error messages and logging
- **Rich Examples**: Python and C++ examples included
- **Testing Support**: CPU backend for local testing without GPUs

---

## Who Should Use TorchComms?

### Target Users

1. **Distributed ML Researchers**
   - Training large language models (LLMs)
   - Large-scale vision models
   - Multi-modal models requiring distributed training

2. **PyTorch Framework Developers**
   - Building distributed training frameworks
   - Implementing custom distributed algorithms
   - Extending PyTorch's distributed capabilities

3. **Cloud/HPC Engineers**
   - Managing large-scale GPU clusters
   - Optimizing distributed training infrastructure
   - Benchmarking communication performance

4. **Framework/Library Developers**
   - Building on top of PyTorch (FSDP, DDP enhancements)
   - Creating custom distributed training libraries
   - Implementing new parallelism strategies

5. **Hardware Vendors**
   - Supporting new GPU types
   - Optimizing for specific network hardware
   - Creating custom backend implementations

### Ideal Use Cases

1. **Large-Scale Model Training**
   - Models with billions of parameters
   - Multi-node, multi-GPU training
   - Gradient synchronization optimization

2. **Heterogeneous Hardware Environments**
   - Mixed NVIDIA and AMD GPU clusters
   - Different network topologies in one cluster
   - CPU fallback for testing

3. **Research Experimentation**
   - Testing new collective algorithms
   - Benchmarking different backends
   - Developing custom communication patterns

4. **Production Distributed Training**
   - Reliable multi-node training
   - Timeout and error handling requirements
   - Performance-critical applications

---

## Main Concepts

### 1. Communicators

A **communicator** is the main object for performing distributed operations:

```python
import torch
from torchcomms import new_comm

# Create a communicator
comm = new_comm(
    backend="ncclx",                    # Backend to use
    device=torch.device("cuda:0"),      # Device for communication
    name="main_comm",                   # Communicator name
    # Optional: custom options
    # options=CommOptions(timeout=timedelta(seconds=30))
)

# Get basic info
rank = comm.get_rank()          # Current process rank (0 to N-1)
world_size = comm.get_size()    # Total number of processes
```

### 2. Reduction Operations

Define how data is combined during collective operations:

```python
from torchcomms import ReduceOp

# Available operations:
ReduceOp.SUM        # Sum all values
ReduceOp.PRODUCT    # Multiply all values
ReduceOp.MIN        # Minimum value
ReduceOp.MAX        # Maximum value
ReduceOp.BAND       # Bitwise AND
ReduceOp.BOR        # Bitwise OR
ReduceOp.BXOR       # Bitwise XOR
ReduceOp.AVG        # Average
```

### 3. Work Objects

Handle asynchronous operation completion:

```python
# Start async operation
work = comm.all_reduce(tensor, ReduceOp.SUM, async_op=True)

# Check if completed (non-blocking)
if work.is_completed():
    print("Communication finished!")

# Wait for completion (blocking)
work.wait()
```

### 4. Communication Options

Control operation behavior with options:

```python
from torchcomms import AllReduceOptions
from datetime import timedelta

# Create options
options = AllReduceOptions(
    timeout=timedelta(seconds=10),
    hints={
        "torchcomm::ncclx::high_priority_stream": "true"
    }
)

# Use in operation
comm.all_reduce(tensor, ReduceOp.SUM, options=options)
```

### 5. Subcommunicators

Create separate communication groups:

```python
# Create subcommunicator with ranks 0-3
sub_comm = comm.split([0, 1, 2, 3], "subgroup")

# Now only ranks 0-3 can communicate via sub_comm
sub_comm.all_reduce(tensor, ReduceOp.SUM)
```

---

## Getting Started

### Installation

#### Option 1: From Source

```bash
# Clone repository
git clone https://github.com/fairinternal/torchcomms.git
cd torchcomms

# Install dependencies
pip install torch pybind11

# Build and install
mkdir build && cd build
cmake .. -DUSE_NCCLX=ON -DUSE_GLOO=ON
make -j
cd ..
pip install -e .
```

#### Option 2: Using Build Script

```bash
python scripts/build.py --backend=ncclx --backend=gloo
```

### Basic Setup

#### Single-Node Multi-GPU

```bash
# Using torchrun (recommended)
torchrun --nproc_per_node=4 my_script.py
```

#### Multi-Node Setup

```bash
# On master node (node 0)
torchrun \
  --nnodes=2 \
  --nproc_per_node=8 \
  --node_rank=0 \
  --rdzv-endpoint="master-hostname:29500" \
  my_script.py

# On worker node (node 1)
torchrun \
  --nnodes=2 \
  --nproc_per_node=8 \
  --node_rank=1 \
  --rdzv-endpoint="master-hostname:29500" \
  my_script.py
```

---

## Usage Examples

### Example 1: Basic Synchronous AllReduce

**File**: `examples/AllReduceSync.py`

```python
#!/usr/bin/env python3
import torch
from torchcomms import new_comm, ReduceOp

def main():
    # Initialize communicator
    device = torch.device("cuda")
    comm = new_comm("ncclx", device, name="main_comm")

    rank = comm.get_rank()
    world_size = comm.get_size()
    device_id = rank % torch.cuda.device_count()
    target_device = torch.device(f"cuda:{device_id}")

    # Create tensor with rank-specific value
    # Rank 0: [1, 1, 1, ...]
    # Rank 1: [2, 2, 2, ...]
    # Rank 2: [3, 3, 3, ...]
    tensor = torch.full(
        (1024,),
        float(rank + 1),
        dtype=torch.float32,
        device=target_device
    )

    print(f"Rank {rank}: Before AllReduce: {tensor[0].item()}")

    # Perform synchronous all-reduce (SUM)
    comm.all_reduce(tensor, ReduceOp.SUM, async_op=False)

    # With 4 ranks, result should be 1+2+3+4 = 10
    torch.cuda.current_stream().synchronize()
    print(f"Rank {rank}: After AllReduce: {tensor[0].item()}")

    # Cleanup
    comm.finalize()

if __name__ == "__main__":
    main()
```

**Run**:
```bash
torchrun --nproc_per_node=4 examples/AllReduceSync.py
```

**Expected Output** (4 GPUs):
```
Rank 0: Before AllReduce: 1.0
Rank 1: Before AllReduce: 2.0
Rank 2: Before AllReduce: 3.0
Rank 3: Before AllReduce: 4.0
Rank 0: After AllReduce: 10.0
Rank 1: After AllReduce: 10.0
Rank 2: After AllReduce: 10.0
Rank 3: After AllReduce: 10.0
```

---

### Example 2: Asynchronous AllReduce with Overlap

**File**: `examples/AllReduceAsync.py`

```python
#!/usr/bin/env python3
import torch
from torchcomms import new_comm, ReduceOp
import time

def main():
    device = torch.device("cuda")
    comm = new_comm("ncclx", device, name="main_comm")

    rank = comm.get_rank()
    device_id = rank % torch.cuda.device_count()
    target_device = torch.device(f"cuda:{device_id}")

    tensor = torch.full(
        (1024,),
        float(rank + 1),
        dtype=torch.float32,
        device=target_device
    )

    # Start async AllReduce - returns immediately!
    work = comm.all_reduce(tensor, ReduceOp.SUM, async_op=True)

    # Do other work while communication happens in background
    print(f"Rank {rank}: Doing other computation...")
    time.sleep(0.01)  # Simulate computation

    # You can check if communication is done
    if not work.is_completed():
        print(f"Rank {rank}: Communication still in progress...")

    # Wait for completion before using the result
    work.wait()

    torch.cuda.current_stream().synchronize()
    print(f"Rank {rank}: Result: {tensor[0].item()}")

    comm.finalize()

if __name__ == "__main__":
    main()
```

**Key Benefits**:
- Communication happens in parallel with computation
- Better GPU utilization
- Reduced training iteration time

---

### Example 3: Point-to-Point Send/Recv (Ring Topology)

**File**: `examples/SendRecvAsync.py`

```python
#!/usr/bin/env python3
import torch
from torchcomms import new_comm

def main():
    device = torch.device("cuda")
    comm = new_comm("ncclx", device, name="main_comm")

    rank = comm.get_rank()
    world_size = comm.get_size()
    device_id = rank % torch.cuda.device_count()
    target_device = torch.device(f"cuda:{device_id}")

    # Create tensors
    send_tensor = torch.full(
        (1024,),
        float(rank),
        dtype=torch.float32,
        device=target_device
    )
    recv_tensor = torch.zeros(1024, dtype=torch.float32, device=target_device)

    # Ring topology: each rank sends to next, receives from previous
    send_rank = (rank + 1) % world_size
    recv_rank = (rank - 1 + world_size) % world_size

    print(f"Rank {rank}: Sending {send_tensor[0].item()} to rank {send_rank}")
    print(f"Rank {rank}: Receiving from rank {recv_rank}")

    # IMPORTANT: Alternate send/recv to avoid deadlock!
    if rank % 2 == 0:
        send_work = comm.send(send_tensor, send_rank, async_op=True)
        recv_work = comm.recv(recv_tensor, recv_rank, async_op=True)
    else:
        recv_work = comm.recv(recv_tensor, recv_rank, async_op=True)
        send_work = comm.send(send_tensor, send_rank, async_op=True)

    # Wait for both operations
    send_work.wait()
    recv_work.wait()

    torch.cuda.current_stream().synchronize()
    print(f"Rank {rank}: Received {recv_tensor[0].item()} from rank {recv_rank}")

    comm.finalize()

if __name__ == "__main__":
    main()
```

**Output** (4 ranks):
```
Rank 0: Received 3.0 from rank 3
Rank 1: Received 0.0 from rank 0
Rank 2: Received 1.0 from rank 1
Rank 3: Received 2.0 from rank 2
```

---

### Example 4: All-Gather Operation

```python
#!/usr/bin/env python3
import torch
from torchcomms import new_comm

def main():
    comm = new_comm("ncclx", torch.device("cuda"))
    rank = comm.get_rank()
    world_size = comm.get_size()

    # Each rank creates a unique tensor
    input_tensor = torch.full(
        (10,),
        float(rank),
        device=torch.device("cuda")
    )

    # Prepare output list for all gathered tensors
    output_tensors = [
        torch.empty_like(input_tensor)
        for _ in range(world_size)
    ]

    # Perform all-gather
    comm.all_gather(output_tensors, input_tensor, async_op=False)

    # Now output_tensors[i] contains the tensor from rank i
    for i, tensor in enumerate(output_tensors):
        print(f"Rank {rank}: Tensor from rank {i}: {tensor[0].item()}")

    comm.finalize()

if __name__ == "__main__":
    main()
```

---

### Example 5: Broadcast Operation

```python
#!/usr/bin/env python3
import torch
from torchcomms import new_comm

def main():
    comm = new_comm("ncclx", torch.device("cuda"))
    rank = comm.get_rank()

    # Create tensor
    if rank == 0:
        # Rank 0 has the data to broadcast
        tensor = torch.full((100,), 42.0, device=torch.device("cuda"))
    else:
        # Other ranks have empty tensor
        tensor = torch.zeros(100, device=torch.device("cuda"))

    print(f"Rank {rank}: Before broadcast: {tensor[0].item()}")

    # Broadcast from rank 0 to all ranks
    comm.broadcast(tensor, root=0, async_op=False)

    print(f"Rank {rank}: After broadcast: {tensor[0].item()}")
    # All ranks now have 42.0

    comm.finalize()

if __name__ == "__main__":
    main()
```

---

### Example 6: CUDA Graphs Integration

```python
#!/usr/bin/env python3
import torch
import torchcomms

def main():
    # Create communicator
    device = torch.device("cuda:0")
    comm = torchcomms.new_comm("ncclx", device)

    tensor = torch.ones(10, device=device) * comm.get_rank()

    # Capture communication operations in a CUDA graph
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        work = comm.all_reduce(
            tensor,
            torchcomms.ReduceOp.SUM,
            async_op=True
        )

    # Replay the graph multiple times for optimized performance
    # Useful in training loops where same communication pattern repeats
    for iteration in range(10):
        graph.replay()
        torch.cuda.current_stream().synchronize()
        print(f"Iteration {iteration}: {tensor[0].item()}")

    comm.finalize()

if __name__ == "__main__":
    main()
```

**Benefits of CUDA Graphs**:
- Reduced kernel launch overhead
- Better performance in training loops
- Efficient repeated communication patterns

---

### Example 7: Object Collectives

```python
#!/usr/bin/env python3
from torchcomms import objcol, new_comm
import torch

def main():
    comm = new_comm("ncclx", torch.device("cuda"))
    rank = comm.get_rank()
    world_size = comm.get_size()

    # Gather arbitrary Python objects (not just tensors!)
    objects_to_gather = [
        ["foo", "bar"],           # Rank 0
        {"key": "value"},         # Rank 1
        42,                       # Rank 2
        (1, 2, 3)                 # Rank 3
    ]

    # Each rank contributes one object
    my_object = objects_to_gather[rank] if rank < len(objects_to_gather) else None

    # Gather all objects to all ranks
    output = [None for _ in range(world_size)]
    objcol.all_gather_object(comm, output, my_object)

    print(f"Rank {rank}: Gathered objects: {output}")

    comm.finalize()

if __name__ == "__main__":
    main()
```

**Use Cases for Object Collectives**:
- Gathering configuration dictionaries
- Collecting metadata from all ranks
- Synchronizing Python state (not GPU tensors)

---

### Example 8: Batch Operations

```python
#!/usr/bin/env python3
import torch
from torchcomms import new_comm

def main():
    comm = new_comm("ncclx", torch.device("cuda"))
    rank = comm.get_rank()

    # Create batch operation
    batch = comm.batch_op_create()

    # Create tensors
    tensor1 = torch.ones(100, device=torch.device("cuda"))
    tensor2 = torch.zeros(100, device=torch.device("cuda"))
    tensor3 = torch.zeros(100, device=torch.device("cuda"))

    if rank == 0:
        # Rank 0 sends to rank 1
        batch.send(tensor1, dst=1)
        batch.send(tensor2, dst=1)
    elif rank == 1:
        # Rank 1 receives from rank 0
        batch.recv(tensor2, src=0)
        batch.recv(tensor3, src=0)

    # Issue all operations in the batch
    work = batch.issue(async_op=True)
    work.wait()

    print(f"Rank {rank}: Batch operations completed")

    comm.finalize()

if __name__ == "__main__":
    main()
```

---

### Example 9: Distributed Training Integration

```python
#!/usr/bin/env python3
import torch
import torch.nn as nn
from torchcomms import new_comm, ReduceOp

class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(100, 10)

    def forward(self, x):
        return self.fc(x)

def main():
    # Initialize communicator
    comm = new_comm("ncclx", torch.device("cuda"))
    rank = comm.get_rank()
    world_size = comm.get_size()

    # Create model and move to GPU
    device = torch.device(f"cuda:{rank % torch.cuda.device_count()}")
    model = SimpleModel().to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    # Training loop
    for epoch in range(5):
        # Create dummy batch (different per rank for data parallelism)
        inputs = torch.randn(32, 100, device=device)
        targets = torch.randint(0, 10, (32,), device=device)

        # Forward pass
        outputs = model(inputs)
        loss = nn.functional.cross_entropy(outputs, targets)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()

        # Synchronize gradients across all ranks
        for param in model.parameters():
            if param.grad is not None:
                # All-reduce gradients (SUM)
                comm.all_reduce(param.grad, ReduceOp.SUM, async_op=False)
                # Average by world size
                param.grad /= world_size

        # Update parameters
        optimizer.step()

        if rank == 0:
            print(f"Epoch {epoch}: Loss = {loss.item():.4f}")

    comm.finalize()
    print(f"Rank {rank}: Training completed!")

if __name__ == "__main__":
    main()
```

**Run**:
```bash
torchrun --nproc_per_node=4 train.py
```

---

## Configuration

### Environment Variables

```bash
# Communicator behavior
export TORCHCOMM_ABORT_ON_ERROR=true        # Abort on timeout/error
export TORCHCOMM_TIMEOUT_SECONDS=30.0       # Default timeout

# Backend loading
export TORCHCOMMS_BACKEND_LIB_PATH_NCCLX=/path/to/libtorchcomms_ncclx.so

# NCCL/NCCLX specific
export NCCL_DEBUG=INFO                      # Enable debugging output
export NCCL_DEBUG_SUBSYS=ALL               # Debug all subsystems
export NCCL_SOCKET_IFNAME=eth0             # Network interface
export NCCL_IB_DISABLE=0                   # Enable InfiniBand
export NCCL_NET_GDR_LEVEL=5                # GPU Direct RDMA level

# Performance tuning
export NCCL_BUFFSIZE=8388608               # Buffer size
export NCCL_NTHREADS=640                   # Number of threads
```

### Build Configuration

```bash
# CMake options
cmake .. \
  -DUSE_NCCL=ON \          # Build NCCL backend
  -DUSE_NCCLX=ON \         # Build NCCLX backend
  -DUSE_GLOO=ON \          # Build Gloo backend
  -DUSE_RCCL=ON \          # Build RCCL backend (AMD)
  -DUSE_RCCLX=ON \         # Build RCCLX backend (AMD)
  -DUSE_SYSTEM_LIBS=1      # Use system libraries
```

### Runtime Configuration

```python
from torchcomms import CommOptions
from datetime import timedelta

# Configure communicator
options = CommOptions(
    abort_process_on_timeout_or_error=True,
    timeout=timedelta(seconds=30),
    name="my_comm",
    hints={
        "backend_specific_option": "value"
    }
)

comm = new_comm("ncclx", device, options=options)
```

---

## Performance Characteristics

### RDMA Transport Performance

Based on benchmarks from `comms/torchcomms/transport/README.md`:

- **Latency**: Sub-microsecond on modern InfiniBand
- **Bandwidth**: 45+ GB/s on 400Gbps NICs (near line-rate)
- **Memory**: Zero-copy GPU-to-GPU transfers
- **CPU Offload**: Hardware-accelerated RDMA operations

### AllReduce Performance

Typical performance on 8x A100 GPUs with NVLink:

| Tensor Size | Latency | Bandwidth |
|-------------|---------|-----------|
| 4 KB        | ~20 µs  | N/A       |
| 1 MB        | ~100 µs | ~80 GB/s  |
| 16 MB       | ~500 µs | ~250 GB/s |
| 128 MB      | ~2 ms   | ~500 GB/s |

### Best Practices for Performance

1. **Use Asynchronous Operations**
   ```python
   work = comm.all_reduce(tensor, ReduceOp.SUM, async_op=True)
   # Do computation
   work.wait()
   ```

2. **Batch Small Operations**
   ```python
   batch = comm.batch_op_create()
   for tensor in tensors:
       batch.send(tensor, dst)
   batch.issue(async_op=True)
   ```

3. **Use CUDA Graphs for Repeated Patterns**
   ```python
   with torch.cuda.graph(graph):
       comm.all_reduce(tensor, ReduceOp.SUM, async_op=True)
   # Replay many times
   ```

4. **Choose Right Backend**
   - NCCLX: NVIDIA GPUs with advanced features
   - RCCLX: AMD GPUs with optimizations
   - Gloo: CPU testing, small-scale work

5. **Configure Network Properly**
   - Use InfiniBand or RoCE for best performance
   - Enable GPU Direct RDMA (`NCCL_NET_GDR_LEVEL=5`)
   - Set appropriate network interface (`NCCL_SOCKET_IFNAME`)

---

## Troubleshooting

### Common Issues

1. **Timeout Errors**
   ```python
   # Increase timeout
   options = CommOptions(timeout=timedelta(seconds=120))
   comm = new_comm("ncclx", device, options=options)
   ```

2. **Backend Not Found**
   ```bash
   # Specify backend library path
   export TORCHCOMMS_BACKEND_LIB_PATH_NCCLX=/path/to/lib.so
   ```

3. **Deadlock in Send/Recv**
   ```python
   # Always alternate send/recv to avoid deadlock
   if rank % 2 == 0:
       send_work = comm.send(...)
       recv_work = comm.recv(...)
   else:
       recv_work = comm.recv(...)
       send_work = comm.send(...)
   ```

4. **Multi-Node Issues**
   ```bash
   # Ensure proper network interface
   export NCCL_SOCKET_IFNAME=eth0  # or ib0 for InfiniBand

   # Enable debug logging
   export NCCL_DEBUG=INFO
   ```

---

## Additional Resources

### Key Files Reference

| Aspect | File Locations |
|--------|----------------|
| **Python API** | `comms/torchcomms/__init__.py`, `_comms.pyi` |
| **Core Interface** | `comms/torchcomms/TorchComm.hpp` |
| **Backend Interface** | `comms/torchcomms/TorchCommBackend.hpp` |
| **Factory** | `comms/torchcomms/TorchCommFactory.cpp` |
| **Examples** | `comms/torchcomms/examples/` |
| **Tests** | `comms/torchcomms/tests/` |
| **Documentation** | `README.md`, `comms/torchcomms/transport/README.md` |

### Further Reading

- **NCCL Documentation**: https://docs.nvidia.com/deeplearning/nccl/
- **PyTorch Distributed**: https://pytorch.org/docs/stable/distributed.html
- **RDMA Overview**: Study InfiniBand and RoCE technologies
- **Collective Algorithms**: Ring-AllReduce, Tree-AllReduce patterns

---

## Summary

TorchComms is a powerful, flexible communications library for PyTorch that provides:

- **Unified API** across NVIDIA, AMD, and CPU backends
- **High Performance** through RDMA, zero-copy transfers, and optimized algorithms
- **Flexibility** with async operations, CUDA graphs, and custom backends
- **Scalability** from single-GPU to massive multi-node clusters
- **Developer-Friendly** with Pythonic interface and comprehensive examples

Whether you're training large language models, conducting distributed ML research, or building the next generation of PyTorch training frameworks, TorchComms provides the communication primitives you need for success.

---

**Happy Distributed Training!**
