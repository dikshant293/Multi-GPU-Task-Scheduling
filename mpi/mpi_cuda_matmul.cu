#include <mpi.h>
#include <cuda_runtime.h>
#include <iostream>

// CUDA kernel to multiply matrices
__global__ void matrixMulKernel(float *a, float *b, float *c, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    float sum = 0.0f;
    if (row < N && col < N) {
        for (int k = 0; k < N; k++) {
            sum += a[row * N + k] * b[k * N + col];
        }
        c[row * N + col] = sum;
    }
}

// Helper function to perform matrix multiplication on the GPU
void matrixMultiply(float *h_a, float *h_b, float *h_c, int numElements) {
    float *d_a, *d_b, *d_c;
    size_t size = numElements * numElements * sizeof(float);

    cudaMalloc(&d_a, size);
    cudaMalloc(&d_b, size);
    cudaMalloc(&d_c, size);

    cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, size, cudaMemcpyHostToDevice);

    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((numElements + threadsPerBlock.x - 1) / threadsPerBlock.x, (numElements + threadsPerBlock.y - 1) / threadsPerBlock.y);
    matrixMulKernel<<<blocksPerGrid, threadsPerBlock>>>(d_a, d_b, d_c, numElements);

    cudaMemcpy(h_c, d_c, size, cudaMemcpyDeviceToHost);

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
}

int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);

    int ndevs = 0;
    cudaError_t error_id = cudaGetDeviceCount(&ndevs);

    int world_size, world_rank;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    printf("world size %d rank %d GPUs %d\n",world_size,world_rank,ndevs);
    int N = 1024;  // Assuming a square matrix of size N x N
    int rows_per_process = N / world_size;
    int numElements = rows_per_process * N;

    float *a, *b, *sub_c;
    float *sub_a = new float[numElements];
    float *sub_b = new float[numElements];
    sub_c = new float[numElements];

    if (world_rank == 0) {
        a = new float[N * N];
        b = new float[N * N];
        // Initialize matrices a and b
        for (int i = 0; i < N * N; i++) {
            a[i] = 1.0;  // Simplified initialization for demonstration
            b[i] = 1.0;
        }
    }

    // Scatter rows of A and B to different MPI processes
    MPI_Scatter(a, numElements, MPI_FLOAT, sub_a, numElements, MPI_FLOAT, 0, MPI_COMM_WORLD);
    MPI_Scatter(b, numElements, MPI_FLOAT, sub_b, numElements, MPI_FLOAT, 0, MPI_COMM_WORLD);

    // Perform matrix multiplication on the GPU
    matrixMultiply(sub_a, sub_b, sub_c, rows_per_process);

    // Gather the resulting submatrices into the final matrix c
    if (world_rank == 0) {
        float *c = new float[N * N];
        MPI_Gather(sub_c, numElements, MPI_FLOAT, c, numElements, MPI_FLOAT, 0, MPI_COMM_WORLD);
        
        // Use matrix c for further processing or output
        delete[] c;
    } else {
        MPI_Gather(sub_c, numElements, MPI_FLOAT, NULL, 0, MPI_FLOAT, 0, MPI_COMM_WORLD);
    }

    delete[] sub_a;
    delete[] sub_b;
    delete[] sub_c;
    if (world_rank == 0) {
        delete[] a;
        delete[] b;
    }

    MPI_Finalize();
    return 0;
}
