#include <mpi.h>
#include <nccl.h>  // Include NCCL header
#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <assert.h>
#include <vector>
#include <iostream>
#include <string>
#include <algorithm>
#include <atomic>
#include <mutex>
#include <thread>
#include <omp.h>
#include <cstdlib>
#include <numeric>
#include <cmath>
#include <ctime>

#if defined(USEOPENMP)
#include <omp.h>
#else
#include <cuda.h>
#include <cuda_runtime.h>
#endif

#if defined(VECTORIZE)
#include <vector_types.h>
#endif

#define CEIL(x, y) (((x) + (y) - 1) / (y))
#define MAX(x, y) (((x) > (y)) ? (x) : (y))
#define MIN(x, y) (((x) < (y)) ? (x) : (y))
#define ABS(x) (((x) < (0)) ? (-x) : (x))
#define MAX_TPB 32

#define EPSILON 1e-4

#define PSIZE 2000

#if not defined(USEOPENMP)
// Kernel for matrix-matrix multiplication
__global__ void multiply_kernel(float *A, float *B, float *C, int M, int N, int K)
{
    for(int i = blockIdx.y * blockDim.y + threadIdx.y;i<M;i+=blockDim.y*gridDim.y){
        for(int j = blockIdx.x * blockDim.x + threadIdx.x;j<N;j+=blockDim.x*gridDim.x)
            {
                float sum = 0.0;

                #if defined(VECTORIZE)
                auto a = reinterpret_cast<float4*>(&A[i * K]);
                auto b = reinterpret_cast<float4*>(&B[j * K]);
                for (int k = 0; k < K/4; k++)
                {
                    sum += a->x*b->x + a->y*b->y + a->z*b->z + a->w*b->w;
                    a++;
                    b++;
                }

                #else
                for (int k = 0; k < K; ++k)
                    sum += A[i * K + k] * B[j * K + k];
                #endif
                C[i * N + j] = sum;
            }
    }
}

__host__ inline cudaError_t checkCuda(cudaError_t status)
{
    if (status != cudaSuccess)
    {
        std::cout << "CUDA Runtime Error: " << cudaGetErrorString(status)
                  << std::endl;
        std::exit(EXIT_FAILURE);
    }
    return status;
}
#endif

void printMatrix(float *mat, int m, int n){
    for(int i=0;i<m;i++){
        for(int j=0;j<n;j++){
            printf("%0.2lf ",mat[i*n+j]);
        }
        printf("\n");
    }
    printf("\n");
}

auto clk = std::chrono::high_resolution_clock::now();

void start_timer(){
    clk = std::chrono::high_resolution_clock::now();
}

void end_timer(std::string func){
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::high_resolution_clock::now() - clk);
    std::cout<<func<<" took "<<1.0e-9 * duration.count()<<" seconds\n";
}

std::vector<int> generateEqualChunkStartIndices(int n, int m) {
    std::vector<int> startIndexes;
    int baseSize = n / m;               // Base size of each chunk
    int remainder = n % m;              // Remainder to be distributed
    int startIndex = 0;

    // Generate starting indices based on uniform chunk sizes
    for (int i = 0; i < m; ++i) {
        startIndexes.push_back(startIndex);
        int currentChunkSize = baseSize + (i < remainder ? 1 : 0);  // Distribute remainder among the first few chunks
        startIndex += currentChunkSize;
    }

    return startIndexes;
}

// Function to calculate chunk sizes from start indices
std::vector<int> calculateChunkSizes(const std::vector<int>& startIndexes, int n) {
    std::vector<int> chunkSizes;
    for (size_t i = 0; i < startIndexes.size(); ++i) {
        if (i == startIndexes.size() - 1) {
            chunkSizes.push_back(n - startIndexes[i]);  // Last chunk goes to the end of the array
        } else {
            chunkSizes.push_back(startIndexes[i + 1] - startIndexes[i]);
        }
    }
    return chunkSizes;
}

void checkMPIError(int status) {
    if (status != MPI_SUCCESS) {
        char errorString[MPI_MAX_ERROR_STRING];
        int lengthOfErrorString;
        MPI_Error_string(status, errorString, &lengthOfErrorString);

        std::cerr << "Error code: " << status << "\n"
                  << "Error description: " << errorString << std::endl;

        MPI_Abort(MPI_COMM_WORLD, status);  // Abort MPI execution
        std::exit(EXIT_FAILURE);            // Terminate the program
    }
}

inline void multiply(float *d_a, float *d_b, float *d_c, int M, int N, int K, int a_items=0, int b_items=0, int c_items=0)
{
    #pragma omp target teams distribute parallel for num_teams(CEIL(M*N,1024)) thread_limit(1024) schedule (static, 1) map(to:d_a[0:a_items],d_b[0:b_items]) map(tofrom:d_c[0:c_items])
    for(int x = 0;x<M*N;x++){
        int ii = x / N, jj = x % N;
        float sum = float();
        #if defined(VECTORIZE)
        auto a = reinterpret_cast<float4*>(&d_a[ii * K]);
        auto b = reinterpret_cast<float4*>(&d_b[jj * K]);
        for (int k = 0; k < K/4; k++)
        {
            sum += a->x*b->x + a->y*b->y + a->z*b->z + a->w*b->w;
            a++;
            b++;
        }

        #else
        for (int kk = 0; kk < K; kk++){
            sum += d_a[ii * K + kk] * d_b[jj * K + kk];
        }
        #endif
        d_c[ii * N + jj] = sum;
    }
}

void transposeMatrix(float* matrix, int m, int n) {
    for (int i = 0; i < m; ++i) {
        for (int j = i + 1; j < n; ++j) {
            std::swap(matrix[i * n + j], matrix[j * n + i]);
        }
    }
}

int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);

    int world_size, world_rank, local_rank;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    int check_result = 0;

    // Get local rank (needed for NCCL communicator)
    const char* local_rank_env = std::getenv("OMPI_COMM_WORLD_LOCAL_RANK");
    if (local_rank_env) {
        local_rank = std::atoi(local_rank_env);
    } else {
        local_rank = world_rank;
    }

    int M = PSIZE, N = PSIZE, K = PSIZE;
    if (argc <= 1)
    {
        printf("Usage bench_works [m] [n] [k]\n");
        printf("Using default parameters\n");
    }
    else
    {
        if (argc > 1)
            M = atoi(argv[1]);
        if (argc > 2)
            N = atoi(argv[2]);
        if (argc > 3)
            K = atoi(argv[3]);
        if (argc > 4)
            check_result = atoi(argv[4]);
    }
    int a_size = M * K, b_size = K * N, c_size = M * N;
    // we spawn total rank number of tasks
    int numRowsPerRank = CEIL(M,world_size);

    std::vector<int> startIndexes = generateEqualChunkStartIndices(M, world_size);
    std::vector<int> chunkSizes = calculateChunkSizes(startIndexes, M);

    // start and end indexes for the current rank
    int start = startIndexes[world_rank], end = (world_rank==world_size-1 ? M : startIndexes[world_rank+1]);
    int nRows = end-start;
    int a_start, b_start, c_start, a_items, b_items, c_items, m, n, k;

    m=nRows; n=N; k=K;
    a_start = start*K; b_start = 0;   c_start = start*N;
    a_items = nRows*K; b_items = K*N; c_items = nRows*N;

    // compute counts and displacements for MPI_Scatterv and MPI_Gatherv
    int *sendcounts_a = new int[world_size];
    int *displs_a = new int[world_size];
    int *recvcounts_c = new int[world_size];
    int *displs_c = new int[world_size];

    for (int i = 0; i < world_size; i++) {
        sendcounts_a[i] = chunkSizes[i] * K; // number of elements to send to process i
        displs_a[i] = startIndexes[i] * K;   // displacement in send buffer
        recvcounts_c[i] = chunkSizes[i] * N;
        displs_c[i] = startIndexes[i] * N;
    }

    // initialize the matrices on rank 0
    float *a,*b,*c;
    if (world_rank == 0) {
        #if defined(USEOPENMP)
        a = (float*)malloc(a_size*sizeof(float));
        b = (float*)malloc(b_size*sizeof(float));
        c = (float*)malloc(c_size*sizeof(float));
        #else
        checkCuda(cudaMallocHost(&a,a_size*sizeof(float)));
        checkCuda(cudaMallocHost(&b,b_size*sizeof(float)));
        checkCuda(cudaMallocHost(&c,c_size*sizeof(float)));
        #endif

        for (int i = 0; i < a_size; i++)
            // a[i] = (float)rand() / RAND_MAX * 2.0 - 1.0;
            a[i] = i%7;

        for (int i = 0; i < b_size; i++)
            // b[i] = (float)rand() / RAND_MAX * 2.0 - 1.0;
            b[i] = 9+i%8;

        for (int i = 0; i < c_size; i++)
            c[i] = 0.0;

        start_timer();

        printf("bench_works [m=%d] [n=%d] [k=%d]\n",M, N, K);
        #if defined(VECTORIZE)
        printf("vectorized\n");
        #else
        printf("non vectorized\n");
        #endif
        #if defined(USEOPENMP)
        printf("using OPENMP target offload\n");
        #else
        printf("using CUDA\n");
        #endif
    }

    // All processes allocate h_a, h_b, h_c
    float *h_a, *h_b, *h_c;

    #if defined(USEOPENMP)
    h_a = (float*)malloc(a_items*sizeof(float));
    h_b = (float*)malloc(b_items*sizeof(float));
    h_c = (float*)malloc(c_items*sizeof(float));
    #else
    checkCuda(cudaMallocHost(&h_a,a_items*sizeof(float)));
    checkCuda(cudaMallocHost(&h_b,b_items*sizeof(float)));
    checkCuda(cudaMallocHost(&h_c,c_items*sizeof(float)));
    #endif

    // Initialize NCCL
    ncclUniqueId ncclId;
    ncclComm_t ncclComm;
    cudaStream_t stream;

    if (world_rank == 0) {
        ncclGetUniqueId(&ncclId);
    }
    checkMPIError(MPI_Bcast(&ncclId, sizeof(ncclId), MPI_BYTE, 0, MPI_COMM_WORLD));
    printf("local rank %d\n",local_rank);
    // checkCuda(cudaSetDevice(local_rank));
    checkCuda(cudaStreamCreate(&stream));
    ncclResult_t ncclStatus = ncclCommInitRank(&ncclComm, world_size, ncclId, world_rank);
    if (ncclStatus != ncclSuccess) {
        printf("NCCL Error: %s\n", ncclGetErrorString(ncclStatus));
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }

    // Device pointers
    float *d_a, *d_b, *d_c;

    // Allocate device memory
    checkCuda(cudaMalloc(&d_a, a_items * sizeof(float)));
    checkCuda(cudaMalloc(&d_b, b_items * sizeof(float)));
    checkCuda(cudaMalloc(&d_c, c_items * sizeof(float)));

    // Copy b from root to all processes using NCCL_Bcast
    if (world_rank == 0) {
        // Copy b to device memory
        checkCuda(cudaMemcpyAsync(d_b, b, b_items * sizeof(float), cudaMemcpyHostToDevice, stream));
    }

    // NCCL Broadcast for d_b
    ncclBroadcast((const void*)d_b, (void*)d_b, b_items, ncclFloat, 0, ncclComm, stream);

    // Copy a chunk of a to all processes using MPI_Scatterv
    if (world_rank == 0) {
        // Copy a to h_a
        memcpy(h_a, a + a_start, a_items * sizeof(float));
    }
    // Scatter h_a to all processes
    checkMPIError(MPI_Scatterv(a, sendcounts_a, displs_a, MPI_FLOAT, h_a, a_items, MPI_FLOAT, 0, MPI_COMM_WORLD));

    // Copy h_a to device
    checkCuda(cudaMemcpyAsync(d_a, h_a, a_items * sizeof(float), cudaMemcpyHostToDevice, stream));

    // Transpose d_b on device (Note: Implement device-side transpose)
    // For simplicity, we can transpose h_b on host and copy it to device
    if (world_rank == 0) {
        checkCuda(cudaMemcpyAsync(h_b, d_b, b_items * sizeof(float), cudaMemcpyDeviceToHost, stream));
        cudaStreamSynchronize(stream);
        transposeMatrix(h_b, K, N);
        checkCuda(cudaMemcpyAsync(d_b, h_b, b_items * sizeof(float), cudaMemcpyHostToDevice, stream));
    }
    // Synchronize NCCL Broadcast
    cudaStreamSynchronize(stream);

    // Synchronize all processes
    checkMPIError(MPI_Barrier(MPI_COMM_WORLD));

    // Now perform the multiplication on device
    #if defined(USEOPENMP)
    // Not applicable since we're using CUDA
    #else // if MPI+CUDA
    int blk_x = MIN(MAX_TPB,n), blk_y = MIN(MAX_TPB,m);
    dim3 blocksPerGrid(CEIL(n,blk_x), CEIL(m,blk_y));
    dim3 threadsPerBlock(blk_x, blk_y); // Assuming width and height are within max threads per block limit

    multiply_kernel<<<blocksPerGrid,threadsPerBlock, 0, stream>>>(d_a,d_b,d_c,m,n,k);
    #endif

    // Copy result back to host
    checkCuda(cudaMemcpyAsync(h_c, d_c, c_items*sizeof(float), cudaMemcpyDeviceToHost, stream));

    // Synchronize stream
    cudaStreamSynchronize(stream);

    // Transpose d_b back if needed (skip if not necessary)

    // Use MPI_Gatherv to gather h_c to c on root process
    checkMPIError(MPI_Gatherv(h_c, c_items, MPI_FLOAT, c, recvcounts_c, displs_c, MPI_FLOAT, 0, MPI_COMM_WORLD));

    // correctness check
    if(world_rank==0){
        end_timer("GPU Multiplication");
        if(check_result){
            float *c_cpu;
            c_cpu = (float*)malloc(c_size*sizeof(float));
            start_timer();
            printf("GPU Done... now checking correctness\n");
            #pragma omp parallel for
            for (int ii = 0; ii < M; ii++)
                for (int jj = 0; jj < N; jj++){
                    c_cpu[ii * N + jj] = 0.0;
                    for (int kk = 0; kk < K; kk++)
                        c_cpu[ii * N + jj] += a[ii * K + kk] * b[kk * N + jj];
                }
            end_timer("CPU multiplication");

            bool flag = true;
            int mismatches = 0;
            for (int i = 0; i < M; i++)
            {
                for (int j = 0; j < N; j++){
                    float x = c[i * N + j], y = c_cpu[i * N + j];
                    if (x != y && ABS((x - y)) > EPSILON)
                    {
                        printf("(%d,%d) : got %lf expected %lf diff %e\n",i,j,x,y,ABS((x - y)));
                        flag = false;
                        mismatches++;
                        break;
                    }
                }
                if (!flag)
                    break;
            }
            printf("Correctness check: %s (mismatches = %d)\n",(flag ? "PASSED" : "FAILED"), mismatches);
            free(c_cpu);
        }
        #if defined(USEOPENMP)
        free(a);free(b);free(c);
        #else
        cudaFreeHost(a);
        cudaFreeHost(b);
        cudaFreeHost(c);
        #endif
    }

    // Clean up
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    cudaStreamDestroy(stream);
    ncclCommDestroy(ncclComm);

    #if defined(USEOPENMP)
    free(h_a);free(h_b);free(h_c);
    #else
    cudaFreeHost(h_a);
    cudaFreeHost(h_b);
    cudaFreeHost(h_c);
    #endif

    // Free allocated arrays for counts and displacements
    delete[] sendcounts_a;
    delete[] displs_a;
    delete[] recvcounts_c;
    delete[] displs_c;

    MPI_Finalize();
    return 0;
}
