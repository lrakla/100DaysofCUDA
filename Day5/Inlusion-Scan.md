## Learnings

- Shared memory from [here](https://developer.nvidia.com/blog/using-shared-memory-cuda-cc/)
    - Shared memory is allocated per thread block, so all threads in the block have access to the same shared memory

- Coalesced Memory 
> Memory coalescing is a technique which allows optimal usage of the global memory bandwidth. That is, when parallel threads running the same instruction access to consecutive locations in the global memory, the most favorable access pattern is achieved.
- Bank Conflicts <br>
When multiple threads in a warp try to access different addresses within the same bank, it causes a "bank conflict," which serializes access and reduces effective bandwidth.
    - How Bank works <br>
        Bank Structure: The entire shared memory is divided into several (typically 32 for modern NVIDIA GPUs) equally-sized memory banks.
        Simultaneous Access: Each bank can service one memory request per cycle independently of other banks.
        Bandwidth Multiplication: If n memory addresses being accessed span b different banks, those b accesses can happen simultaneously, giving you b times the bandwidth of a single bank.

## FAQ
1. What if you need multiple dynamically sized arrays in a single kernel? <br>
You must declare a single extern unsized array as before, and use pointers into it to divide it into multiple arrays, as in the following excerpt.

```cpp
extern __shared__ int s[];
int *integerData = s;                        // nI ints
float *floatData = (float*)&integerData[nI]; // nFfloats
char *charData = (char*)&floatData[nF];      // nC chars
In the kernel launch, specify the total shared memory needed, as in the following.

myKernel<<<gridSize, blockSize, nI*sizeof(int)+nF*sizeof(float)+nC*sizeof(char)>>>(...);
```

2. How to call CUDA kernels from within each other? 
You can't directly call another CUDA kernel from within a kernel in standard CUDA programming. This is a fundamental limitation of the CUDA execution model. Here's why and some alternatives:

Why Direct Kernel-to-Kernel Calls Aren't Possible

1. **Execution Model**: CUDA kernels are launched by the host (CPU) to run on the device (GPU). The GPU execution model doesn't natively support launching new work from within executing kernels in standard CUDA.

2. **Synchronization Issues**: Kernels run asynchronously with respect to the host, and there's no built-in mechanism for one kernel to wait for another kernel it might launch.

Alternatives and Solutions

1. **Dynamic Parallelism** (Compute Capability 3.5+)
   This is the closest thing to calling a kernel from another kernel:

   ```cuda
   __global__ void childKernel() {
       // Child kernel code
   }
   
   __global__ void parentKernel() {
       // Parent kernel code
       
       // Launch child kernel
       childKernel<<<gridSize, blockSize>>>();
       
       // Need to synchronize to wait for child kernel
       cudaDeviceSynchronize();
       
       // Continue with parent kernel
   }
   ```

   Requirements:
   - Device must support compute capability 3.5 or higher
   - Need to compile with `-rdc=true` flag
   - Need to use separate compilation and linking

2. **Sequential Kernel Launches from Host**
   The most common approach is to break your work into multiple kernels and launch them sequentially from the host:

   ```cuda
   // On the host side
   kernel1<<<grid, block>>>(data);
   kernel2<<<grid, block>>>(data);
   ```

3. **Persistent Threads Pattern**
   Create a kernel that runs continuously and processes work items from a queue:

   ```cuda
   __global__ void persistentKernel(WorkQueue* queue) {
       while (true) {
           WorkItem item = queue->dequeue();
           if (item.type == TERMINATE) break;
           
           // Process work item
           if (item.type == OPERATION_1) {
               // Do operation 1
           } else if (item.type == OPERATION_2) {
               // Do operation 2
           }
       }
   }
   ```

4. **Using Cooperative Groups** (newer CUDA versions)
   This provides more flexibility for synchronization and control within kernels.

### Best Practices

- Break complex operations into separate kernels and manage the flow from the host
- Use streams for better concurrency if kernels can run in parallel
- Consider using unified memory (if available) to simplify data management between kernel calls
- For newer GPUs, look into CUDA Graphs for efficient repeated execution of kernel sequences

Dynamic Parallelism is the only official way to launch a kernel from within another kernel, but it comes with performance considerations and is not supported on all devices.
### Resources
- https://nyu-cds.github.io/python-gpu/02-cuda/ (great article for basics)
- 