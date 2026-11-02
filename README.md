# Phase 1: Foundations of Deep Learning & PyTorch Internals

## Phase Overview & Systems Objective
Before scaling models across thousands of GPUs or optimizing network communication fabrics, you must understand how high-level deep learning code translates into raw compute and memory traffic at the hardware level. 

Coming from a silicon/physical design background, think of this phase as moving from **transistor-level gate logic up to microarchitecture RTL execution**. You will demystify how automatic differentiation builds state machines (computational graphs), how memory is laid out in C++ and managed by PyTorch's dynamic allocator, and how custom compute kernels interact with GPU execution units (warps and thread blocks).

---

## Month-by-Month Breakdown & Deliverables

### Month 1: Deep Learning Fundamentals & Tensor Mechanics
* **Core Focus:** Automatic differentiation (`autograd`) mechanics, computational graph construction as Directed Acyclic Graphs (DAGs), and low-level PyTorch tensor memory layouts (strides, storage offsets, and contiguous vs. non-contiguous memory).
* **What You Will Do:**
  1. Build a custom scalar/tensor `autograd` engine from scratch in Python (`engine.py`) featuring dynamic graph traversal, topological sorting, and reverse-mode chain rule propagation.
  2. Write tensor experimentation suites inspecting `.stride()`, `.storage_offset()`, and memory copying behaviors when executing `.view()` vs. `.reshape()`.
* **Deliverable:** A modular, tested custom backpropagation engine and memory-efficient tensor manipulation suite.

### Month 2: Profiling & Memory Management
* **Core Focus:** GPU memory hierarchy (HBM vs. SRAM, L2 cache), activation memory footprints, parameter weight memory, and the internal mechanics of PyTorch’s Caching Allocator (memory pools, fragmentation mitigation).
* **What You Will Do:**
  1. Profile a baseline transformer model using PyTorch Profiler, `torch.cuda.memory`, and PyTorch Memory Snapshots.
  2. Diagnose and isolate root causes for Out-Of-Memory (OOM) conditions and memory fragmentation during the backward pass.
* **Deliverable:** A comprehensive profiling report diagnosing and fixing memory bottlenecks and activation footprint leaks in a baseline model.

### Month 3: Custom CUDA Kernels & Operator Optimization
* **Core Focus:** Bridging hardware thinking to software by mapping compute to GPU threads (thread blocks, warps, shared memory tiling, and memory coalescing). Analyzing memory-bandwidth limitations versus compute-bound operations (the Roofline Model).
* **What You WillDo:**
  1. Write custom CUDA/Triton kernels for core reduction or element-wise operations (e.g., LayerNorm or Softmax).
  2. Benchmark your custom operator speedups against standard PyTorch implementations.
* **Deliverable:** An optimized custom operator kernel complete with a hardware throughput benchmark analysis.

---

## How to Track Your Progress in Git
As you complete each month, organize your code and documentation inside your directory structure like this:
```text
phase1_foundations/
├── month01_autograd/       # Custom autograd engine & tensor stride experiments
├── month02_profiling/      # Profiler outputs, memory snapshots, & OOM fix scripts
└── month03_kernels/        # Custom CUDA/Triton kernels & benchmark scripts
```
