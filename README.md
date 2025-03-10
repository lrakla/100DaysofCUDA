# 100DaysofCUDA
Consistently coding in CUDA for 100 days

| Day | Done |
| -------- | -------- | 
| 1  | Kernel to add 1-D vectors, profiled it using `nvprof` & `ncu`  | 
| 2  | 2-D Matrix Addition  | 

## Execution Environment
I run these codes on Google Colab which has the `nvcc` compiler (as part of the CUDA toolkit)

To run any of these codes in Colab, in a new code block paste
```cu
%%writefile <filename>.cu
------------
Paste file here
------------
```