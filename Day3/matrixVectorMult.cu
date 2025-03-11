#include <iostream>

__global__ void matrixVectMult(float *mat, float *vect, float *res, int N)
{
    int i = blockDim.x*blockIdx.x + threadIdx.x;
    float sum = 0.0f;
    if(i < N)
    {
        for(int j = 0; j < N; j++)
        {
            sum += mat[i*N + j] * vect[j];
        }
        res[i] = sum;
    }
}


int main()
{
    const int N = 20;
    float *mat, *vect, *res;

    mat = (float *)malloc(N*N*sizeof(float));
    vect = (float *)malloc(N*sizeof(float));
    res = (float *)malloc(N*sizeof(float));

    for(int i = 0; i < N; i++)
    {
        for(int j =0; j< N; j++)
        {
            mat[i*N + j] = 1.0f;
        }
    vect[i] = 2.0f;
    res[i] = 0.0f;
    }
    
    float *mat_d, *vect_d, *res_d;
    cudaMalloc((void**)&mat_d, N*N*sizeof(float));
    cudaMalloc((void**)&vect_d, N*sizeof(float));
    cudaMalloc((void**)&res_d, N*sizeof(float));

    cudaMemcpy(mat_d, mat, N*N*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(vect_d, vect, N*sizeof(float), cudaMemcpyHostToDevice);

    dim3 blockDim(16);
    dim3 gridDim(ceil(N/16.0f)); // Be careful about integer division here
    
    matrixVectMult<<<gridDim,blockDim>>>(mat_d,vect_d,res_d,N);
    cudaDeviceSynchronize();

    cudaMemcpy(res, res_d, N*sizeof(float), cudaMemcpyDeviceToHost);

    // verify C is 3.0
    for(int i = 0; i < N; i++)
    {
            if(res[i] != 2.0f * N)
            {
                std::cout << "Error: mismatch at position " << i << std::endl;
                break;
            }
        
    }

    std::cout << "Matrix vector multiplication completed successfully!" << std::endl;


    cudaFree(mat_d);
    cudaFree(vect_d);
    cudaFree(res_d);

}