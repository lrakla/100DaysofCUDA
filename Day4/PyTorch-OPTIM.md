## PyTorch GPU Optimizations

Today I wanted to switch gears and dive into how GPU optimizations can be performed using the PyTorch API. 

 

## Contents

1. [Introduction to PyTorch GPU Optimizations](#introduction-to-pytorch-gpu-optimizations)
2. [Memory Management](#memory-management)
   - [Tensor Memory Layout](#tensor-memory-layout)
   - [Memory Pinning](#memory-pinning)
3. [Computation Optimizations](#computation-optimizations)
   - [Mixed Precision Training](#mixed-precision-training)
   - [Tensor Cores Usage](#tensor-cores-usage)
4. [Data Loading](#data-loading)
   - [GPU Direct Access](#gpu-direct-access)
   - [Prefetching](#prefetching)
5. [Best Practices](#best-practices)
   - [Batch Size Optimization](#batch-size-optimization)
   - [Model Parallelism](#model-parallelism)

## 

#### 2.1 [Memory Pinning](#introduction-to-pytorch-gpu-optimizations)
[This](https://www.sciencedirect.com/topics/computer-science/pinned-host-memory) article explains what is pinned memory: 
>When memory is allocated for variables that reside on the host, pageable memory is used by default. Pageable memory can be swapped out to disk to allow the program to use more memory than is available in RAM on the host system. When data is transferred between the host and the device, the direct memory access (DMA) engine on the GPU must target page-locked or pinned host memory. Pinned memory cannot be swapped out and is therefore always available for such transfers. To accommodate data transfers from pageable host memory to the GPU, the host operating system first allocates a temporary pinned host buffer, copies the data to the pinned buffer, and then transfers the data to the device, as illustrated in Figure 3.1. The pinned memory buffer may be smaller than the pageable memory holding the host data, in which case the transfer occurs in multiple stages. Pinned memory buffers are similarly used with transfers from the device to the host. The cost of the transfer between pageable memory and pinned host buffer can be avoided if we declare the host arrays to use pinned memory.

- PyTorch [blog](https://pytorch.org/tutorials/intermediate/pinmem_nonblock.html) on pinned memory.


### Resources
1. https://engineering.purdue.edu/~smidkiff/ece563/NVidiaGPUTeachingToolkit/Mod14DataXfer/Mod14DataXfer.pdf
2. https://pytorch.org/tutorials/recipes/recipes/tuning_guide.html
