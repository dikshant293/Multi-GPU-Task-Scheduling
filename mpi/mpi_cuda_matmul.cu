#include <mpi.h>
#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <cuda.h>
#include <cuda_runtime.h>
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

#define CEIL(x, y) (((x) + (y) - 1) / (y))
#define MAX(x, y) (((x) > (y)) ? (x) : (y))
#define MIN(x, y) (((x) < (y)) ? (x) : (y))
#define ABS(x) (((x) < (0)) ? (-x) : (x))
#define MAX_TPB 32

#define EPSILON 1e-4

#define PSIZE 2000

// Kernel for matrix-matrix multiplication
__global__ void multiply_kernel(float *A, float *B, float *C, int M, int N, int K)
{
    for(int i = blockIdx.y * blockDim.y + threadIdx.y;i<M;i+=blockDim.y*gridDim.y){
        for(int j = blockIdx.x * blockDim.x + threadIdx.x;j<N;j+=blockDim.x*gridDim.x)
    // if(i < M and j < N)
            {
                // atomicAdd(&d_counter, 1);
                float sum = 0.0;
                
                #if defined(VECTORIZE)
                auto a = reinterpret_cast<float4*>(&A[i * K]);
                auto b = reinterpret_cast<float4*>(&B[j * K]);
                // printf("check %d %d (%p %p) (%p %p) %d\n",i*K*4,j*K*4,a,&A[i * K],b,&B[j * K],K/4);
                for (int k = 0; k < K/4; k++)
                {
                    // printf("before\n");
                    // auto a = a_4[k], b = b_4[k];
                    // printf("%f,%f,%f,%f %f,%f,%f,%f\n",a.w,a.x,a.y,a.z,b.w,b.x,b.y,b.z);
                    sum += a->x*b->x + a->y*b->y + a->z*b->z + a->w*b->w;
                    // printf("(%f,%f)\n",a->w,b->w);
                    a++;
                    b++;
                }
                
                #else
                for (int k = 0; k < K; ++k)
                    sum += A[i * K + k] * B[j * K + k];
                #endif
                C[i * N + j] = sum;

                
            // printf("\n-------------\n");
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


void transposeMatrix(float* matrix, int m, int n) {
    for (int i = 0; i < m; ++i) {
        for (int j = i + 1; j < n; ++j) {
            std::swap(matrix[i * n + j], matrix[j * n + i]);
        }
    }
}

int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);

    int ndevs = 0;
    cudaError_t error_id = cudaGetDeviceCount(&ndevs);

    int world_size, world_rank;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    int check_result = 0;
    // printf("world size %d rank %d GPUs %d\n",world_size,world_rank,ndevs);

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

    int numRowsPerRank = CEIL(M,world_size);
    

    
    std::vector<int> startIndexes = generateEqualChunkStartIndices(M, world_size);;
    std::vector<int> chunkSizes = calculateChunkSizes(startIndexes, M);

    cudaSetDevice(world_rank%ndevs);
    checkMPIError(MPI_Barrier(MPI_COMM_WORLD));
        
    int start = startIndexes[world_rank], end = (world_rank==world_size-1 ? M : startIndexes[world_rank+1]);
    int nRows = end-start;
    int a_start, b_start, c_start, a_items, b_items, c_items, m, n, k;
    
    m=nRows; n=N; k=K;
    a_start = start*K; b_start = 0;   c_start = start*N;
    a_items = nRows*K; b_items = K*N; c_items = nRows*N;
    MPI_Status stat;

    float *a,*b,*c;
    if (world_rank == 0) {

        checkCuda(cudaMallocHost(&a,a_size*sizeof(float)));
        checkCuda(cudaMallocHost(&b,b_size*sizeof(float)));
        checkCuda(cudaMallocHost(&c,c_size*sizeof(float)));

        // initialize

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
        #endif
    }


    float *h_a, *h_b, *h_c;
    if(world_rank==0){
        h_a = a+a_start;
        h_b = b+b_start;
        h_c = c+c_start;
        for(int i=1;i<world_size;i++){
            int send_start = startIndexes[i], send_end = (i==world_size-1 ? M : startIndexes[i+1]);
            int send_nRows = send_end-send_start;
            checkMPIError(MPI_Send(a+send_start*K   ,send_nRows*K   ,MPI_FLOAT,i,1,MPI_COMM_WORLD));
            checkMPIError(MPI_Send(b+0              ,K*N            ,MPI_FLOAT,i,2,MPI_COMM_WORLD));
            checkMPIError(MPI_Send(c+send_start*N   ,send_nRows*N   ,MPI_FLOAT,i,3,MPI_COMM_WORLD));
        }
        
    }
    else{
        
        checkCuda(cudaMallocHost(&h_a,a_items*sizeof(float)));
        checkCuda(cudaMallocHost(&h_b,b_items*sizeof(float)));
        checkCuda(cudaMallocHost(&h_c,c_items*sizeof(float)));

        checkMPIError(MPI_Recv(h_a,a_items,MPI_FLOAT,0,1,MPI_COMM_WORLD,&stat));
        checkMPIError(MPI_Recv(h_b,b_items,MPI_FLOAT,0,2,MPI_COMM_WORLD,&stat));
        checkMPIError(MPI_Recv(h_c,c_items,MPI_FLOAT,0,3,MPI_COMM_WORLD,&stat));
    }
    
    // checkMPIError(MPI_Barrier(MPI_COMM_WORLD));
    // int p = 2;
    // if(world_rank==p){
    //     printMatrix(h_a,m,k);
    // }


    transposeMatrix(h_b,K,N);

    float *d_a,*d_b,*d_c;

    cudaMalloc(&d_a,a_items*sizeof(float));
    cudaMemcpy(d_a,h_a,a_items*sizeof(float),cudaMemcpyHostToDevice);
    
    cudaMalloc(&d_b,b_items*sizeof(float));
    cudaMemcpy(d_b,h_b,b_items*sizeof(float),cudaMemcpyHostToDevice);
    
    cudaMalloc(&d_c,c_items*sizeof(float));
    cudaMemcpy(d_c,h_c,c_items*sizeof(float),cudaMemcpyHostToDevice);

    int blk_x = min(MAX_TPB,n), blk_y = min(MAX_TPB,m);
    dim3 blocksPerGrid(CEIL(n,blk_x), CEIL(m,blk_y));
    dim3 threadsPerBlock(blk_x, blk_y); // Assuming width and height are within max threads per block limit 

    multiply_kernel<<<blocksPerGrid,threadsPerBlock>>>(d_a,d_b,d_c,m,n,k);

    cudaMemcpy(h_c,d_c,c_items*sizeof(float),cudaMemcpyDeviceToHost);

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    cudaDeviceSynchronize();

    transposeMatrix(h_b,N,K);
    
    // int rank = 0;
    // while (rank < world_size) {
    //     if (world_rank == rank) {
    //         printf("rank %d startrow %d endrow %d a_start %d b_start %d c_start %d\n",world_rank,start,end,a_start,b_start,c_start);
    //         printf ("Array printed by rank: %d\n", world_rank);
    //         printMatrix(h_c,m,n);
    //         fflush(stdout);
    //     }
    //     rank++;
    //     MPI_Barrier(MPI_COMM_WORLD);
    // }

    if(world_rank==0){
        h_a = a+a_start;
        h_b = b+b_start;
        h_c = c+c_start;
        for(int i=1;i<world_size;i++){
            int recv_start = startIndexes[i], recv_end = (i==world_size-1 ? M : startIndexes[i+1]);
            int recv_nRows = recv_end-recv_start;
            checkMPIError(MPI_Recv(c+recv_start*N, recv_nRows*N,MPI_FLOAT,i,4,MPI_COMM_WORLD,&stat));
        }
    }
    else{
        checkMPIError(MPI_Send(h_c,c_items,MPI_FLOAT,0,4,MPI_COMM_WORLD));
    }

    if(world_rank==0){
        end_timer("GPU Multiplication");
        if(check_result){
            float *c_cpu;
            // checkCuda(cudaMallocHost(&c_cpu,c_size*sizeof(float)));
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

            // printMatrix(c_cpu,M,N);

            bool flag = true;
            int mismatches = 0;
            for (int i = 0; i < M; i++)
            {
                for (int j = 0; j < N; j++){
                    float x = c[i * N + j], y = c_cpu[i * N + j];
                    if (x != y && ABS((x - y)) > EPSILON) // data_type precision comparision upto 10^-6 for types like doubles
                    {
                        printf("(%d,%d) : got %lf expected %lf diff %e\n",i,j,x,y,ABS((x - y)));
                        flag = false;
                        mismatches++;
                        // break;
                    }
                }
                // if (!flag)
                //     break;
            }
            printf("Correctness check: %s (mismatches = %d)\n",(flag ? "PASSED" : "FAILED"), mismatches);
            cudaFreeHost(c_cpu);
        }
        cudaFreeHost(a);
        cudaFreeHost(b);
        cudaFreeHost(c);
    }
    else{
        cudaFreeHost(h_a);
        cudaFreeHost(h_b);
        cudaFreeHost(h_c);
    }

    MPI_Finalize();
    return 0;
}
