#include <stdio.h>

__global__ 
void partialSumKernel(int *input, int *output, int n) {
    // Shared memory - FIXED: don't declare array size here
    extern __shared__ int sharedMemory[];
    int tid = threadIdx.x;
    int index = blockIdx.x * blockDim.x * 2 + tid; // each thread processed 2 elements
    
    if (index + blockDim.x < n) {
        sharedMemory[tid] = input[index] + input[index + blockDim.x];
    } else if (index < n) {
        sharedMemory[tid] = input[index];
    } else {
        sharedMemory[tid] = 0;
    }
    __syncthreads();
    
    // Perform inclusive scan in shared memory
    for (int stride = 1; stride < blockDim.x; stride *= 2) {
        int temp = 0;
        if (tid >= stride) {
            temp = sharedMemory[tid - stride];
        }
        __syncthreads();
        if (tid >= stride) {
            sharedMemory[tid] += temp;
        }
        __syncthreads();
    }
    
    // FIXED: Only write to output if in bounds
    if (index < n) {
        output[index] = sharedMemory[tid];
    }
}

//Kernel to make it global scan
__global__ void completePrefixSumKernel(int *partialSums, int *output, int n) { // FIXED: Added n parameter
    // Get the block total for each previous block
    extern __shared__ int blockTotals[];
    
    int tid = threadIdx.x;
    int index = blockIdx.x * blockDim.x + tid;
    
    // First, load the last element of each previous block into shared memory
    if (tid < blockIdx.x) {
        blockTotals[tid] = partialSums[(tid + 1) * blockDim.x - 1];
    } else {
        blockTotals[tid] = 0;
    }
    __syncthreads();
    
    // Perform exclusive scan on block totals in shared memory
    for (int stride = 1; stride < blockDim.x; stride *= 2) {
        int temp = 0;
        if (tid >= stride) {
            temp = blockTotals[tid - stride];
        }
        __syncthreads();
        if (tid >= stride) {
            blockTotals[tid] += temp;
        }
        __syncthreads();
    }
    
    // Add the appropriate block total to each element
    if (index < n) {
        int blockSum = 0;
        if (blockIdx.x > 0) {
            blockSum = blockTotals[blockIdx.x - 1];
        }
        output[index] = partialSums[index] + blockSum;
    }
}

int main() {
    const int N = 16;
    const int blockSize = 4; // FIXED: Changed to 4 for better grid/block organization
    const int gridSize = (N + blockSize - 1) / blockSize; // FIXED: Proper grid size calculation

    int h_input[N] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16};
    int h_output[N];
    int h_expected[N] = {1, 3, 6, 10, 15, 21, 28, 36, 45, 55, 66, 78, 91, 105, 120, 136}; // Added for verification

    int *d_input, *d_output, *d_finalOutput;
    size_t size = N * sizeof(int);

    cudaMalloc(&d_input, size);
    cudaMalloc(&d_output, size);
    cudaMalloc(&d_finalOutput, size);
    cudaMemcpy(d_input, h_input, size, cudaMemcpyHostToDevice);

    // For partialSumKernel, each thread processes 2 elements
    int partialGridSize = (N + (blockSize * 2) - 1) / (blockSize * 2);
    partialSumKernel<<<partialGridSize, blockSize, blockSize * sizeof(int)>>>(d_input, d_output, N);

    completePrefixSumKernel<<<gridSize, blockSize, blockSize * sizeof(int)>>>(d_output, d_finalOutput, N);
    
    // Check for kernel launch errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("Kernel launch error: %s\n", cudaGetErrorString(err));
        return -1;
    }
    
    cudaMemcpy(h_output, d_finalOutput, size, cudaMemcpyDeviceToHost);
 
    printf("Input: ");
    for (int i = 0; i < N; i++) {
        printf("%d ", h_input[i]);
    }
    printf("\nOutput: ");
    for (int i = 0; i < N; i++) {
        printf("%d ", h_output[i]);
    }
    printf("\nExpected: ");
    for (int i = 0; i < N; i++) {
        printf("%d ", h_expected[i]);
    }
    printf("\n");

    // Verify results
    int correct = 1;
    for (int i = 0; i < N; i++) {
        if (h_output[i] != h_expected[i]) {
            printf("Mismatch at index %d: got %d, expected %d\n", i, h_output[i], h_expected[i]);
            correct = 0;
        }
    }
    
    if (correct) {
        printf("Prefix sum calculation is correct!\n");
    } else {
        printf("Prefix sum calculation is incorrect.\n");
    }

    cudaFree(d_input);
    cudaFree(d_output);
    cudaFree(d_finalOutput);

    return 0;
}