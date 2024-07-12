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

#define BLOCK_SIZE 4
#define MAX_TPB 32

#define MM
#define PSIZE 2000

#define EPSILON 1e-4

// #define SCHED_ROUNDROBIN
// #define SCHED_DYNAMIC
// #define SCHED_DYNAMIC2
// #define SCHED_RANDOM
// #define SCHED_ADAPTIVE
// #define SCHED_ADAPTIVE2

// using data_type = float;

std::mutex mtx;
// Define the global variable
// __device__ int d_counter = 0;

// #define USEOPENMP
// #define PRE_TRANSFER

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

__host__ inline unsigned gpu_scheduler_static_rr(int taskID, int ngpus)
{
    return taskID % ngpus;
}

__host__ inline unsigned gpu_scheduler_dynamic_ad(unsigned long *gpuLoad, int ngpus, int taskWeight)
{
    short looking = 1;
    unsigned chosen;
    while (looking)
    {
        unsigned occ_i;
        unsigned long load;
        unsigned long min_load = ULLONG_MAX;
        for (unsigned i = 0; i < ngpus; i++)
        {
#pragma omp atomic read
            load = gpuLoad[i];
            if (load < min_load)
            {
                min_load = load;
                occ_i = i;
            }
        }
        chosen = occ_i;
#pragma omp atomic
        gpuLoad[chosen] += taskWeight;
        looking = 0;
        break;
    }
    return chosen;
}

// This version avoids all CPU threads finding the same GPU greedily (and therefore overloading that GPU)
__host__ inline unsigned gpu_scheduler_dynamic_ad2(unsigned long *gpuLoad, int ngpus, int taskWeight)
{
    short looking = 1;
    unsigned chosen;
    while (looking)
    {
        unsigned long load;
        unsigned long min_load = ULLONG_MAX;

#pragma omp critical
        {
            for (unsigned i = 0; i < ngpus; i++)
            {
                load = gpuLoad[i];
                if (load < min_load)
                {
                    min_load = load;
                    chosen = i;
                }
            }
            gpuLoad[chosen] += taskWeight;
        }
        looking = 0;
        break;
    }
    return chosen;
}

__host__ inline unsigned gpu_scheduler_dynamic_random(unsigned *occupancies, int ngpus)
{
    const unsigned chosen = rand() % ngpus;
#pragma omp atomic
    occupancies[chosen]++;
    return chosen;
}

__host__ inline unsigned gpu_scheduler_dynamic_occ2(unsigned *occupancies, int ngpus)
{
    int chosen = -1;
    while (chosen == -1)
    {
        for (unsigned i = 0; i < ngpus; i++)
        {
#pragma omp critical
            {
                if (occupancies[i] == 0)
                {
                    occupancies[i]++;
                    chosen = i;
                }
            }
            if (chosen > -1)
                break;
        }
    }
    return chosen;
}

__host__ inline unsigned gpu_scheduler_dynamic_occ(unsigned *occupancies, int ngpus)
{
    short looking = 1;
    unsigned chosen;
    while (looking)
    {
        for (unsigned i = 0; i < ngpus; i++)
        {
            // But really, this should be a single atomic compare-and-swap
            unsigned occ_i;
#pragma omp atomic read
            occ_i = occupancies[i];
            if (occ_i == 0)
            {
                chosen = i;
#pragma omp atomic
                occupancies[chosen]++;
                looking = 0;
                break;
            }
        }
    }
    return chosen;
}

void transposeMatrix(float* matrix, int m, int n) {
    for (int i = 0; i < m; ++i) {
        for (int j = i + 1; j < n; ++j) {
            std::swap(matrix[i * n + j], matrix[j * n + i]);
        }
    }
}

// Kernel for matrix-matrix multiplication
__global__ void multiply_kernel(float *A, float *B, float *C, int rowStart, int M, int N, int K)
{
    // int i = blockIdx.y * blockDim.y + threadIdx.y;
    // int j = blockIdx.x * blockDim.x + threadIdx.x;
    // // i += rowStart;
    // printf("i inc = %d j inc = %d\n",blockDim.y*gridDim.y,blockDim.x*gridDim.x);
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

__global__ void gpu_check(int i, int d){
    printf("task %d on GPU %d done\n", i, d);
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

__global__ void printMatrixKernel(float* matrix, int width, int height) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    int idy = threadIdx.y + blockIdx.y * blockDim.y;

    if (idx < width && idy < height) {
        printf("Element at [%d, %d]: %f\n", idy, idx, matrix[idy * width + idx]);
    }
}

void printMatrixGPU(float *mat, int m, int n){
    dim3 blocksPerGrid((n+min(MAX_TPB,n)-1)/min(MAX_TPB,n), (m+min(MAX_TPB,m)-1)/min(MAX_TPB,m));
    dim3 threadsPerBlock(min(MAX_TPB,n), min(MAX_TPB,m)); // Assuming width and height are within max threads per block limit 
    printMatrixKernel<<<blocksPerGrid, threadsPerBlock>>>(mat, n, m);
    // Wait for GPU to finish before accessing on host
    cudaDeviceSynchronize();
}

auto clk = std::chrono::high_resolution_clock::now();

void start_timer(){
    clk = std::chrono::high_resolution_clock::now();
}

void end_timer(std::string func){
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::high_resolution_clock::now() - clk);
    std::cout<<func<<" took "<<1.0e-9 * duration.count()<<" seconds\n";
}

void joinThreads(std::vector<std::thread> &threads){
    #if not defined(USEOPENMP)
    for (auto &thread: threads){
        thread.join();
    }
    threads.clear();
    #endif
}

// Function to calculate mean of chunk sizes
double calculateMean(const std::vector<int>& chunkSizes) {
    double sum = 0.0;
    for (int size : chunkSizes) {
        sum += size;
    }
    std::cout << "Mean: " << sum / chunkSizes.size() << "\t";
    return sum / chunkSizes.size();
}

// Function to calculate standard deviation of chunk sizes
double calculateStandardDeviation(const std::vector<int>& chunkSizes, double mean) {
    double sum = 0.0;
    for (int size : chunkSizes) {
        sum += std::pow(size - mean, 2);
    }
    std::cout << "Standard deviation: " << std::sqrt(sum / chunkSizes.size()) << std::endl;
    return std::sqrt(sum / chunkSizes.size());
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

std::vector<int> generateUniformChunkStartIndices(int n, int m) {
    std::vector<int> chunkSizes(m, 1); // Start each chunk with at least one element
    int remainingElements = n - m;    // Elements left after giving 1 to each chunk

    srand(101);
    // Distribute the remaining elements randomly
    for (int i = 0; i < remainingElements; ++i) {
        int chunkIndex = rand() % m;
        chunkSizes[chunkIndex]++;
    }

    // Calculate the starting indices
    std::vector<int> startIndexes(m);
    std::partial_sum(chunkSizes.begin(), chunkSizes.end() - 1, startIndexes.begin() + 1);

    return startIndexes;
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


std::vector<int> generateRandomChunkStartIndices(int n, int m) {
    std::vector<int> chunkSizes;
    std::vector<int> startIndexes;
    int totalSize = 0;

    // Seed the random number generator
    srand(101);

    // Generate random chunk sizes
    for (int i = 0; i < m; ++i) {
        if (i == m - 1) {
            chunkSizes.push_back(n - totalSize); // Last chunk takes the remaining elements
        } else {
            int remaining = n - totalSize - (m - i - 1); // Ensure space for at least 1 element per remaining chunk
            int chunkSize = 1 + rand() % remaining;
            chunkSizes.push_back(chunkSize);
            totalSize += chunkSize;
        }
    }

    // Calculate starting indices from chunk sizes
    int startIndex = 0;
    for (int size : chunkSizes) {
        startIndexes.push_back(startIndex);
        startIndex += size;
    }

    return startIndexes;
}

int main(int argc, char **argv)
{
    int ndevs = 0;
    cudaError_t error_id = cudaGetDeviceCount(&ndevs);

    if (error_id != cudaSuccess)
    {
        printf("cudaGetDeviceCount returned %d\n-> %s\n", (int)error_id, cudaGetErrorString(error_id));
        printf("Result = FAIL\n");
        return 1;
    }

    // Output the number of GPUs
    printf("Number of GPUs available: %d\n", ndevs);
    int *devices = (int *)calloc(ndevs, sizeof(*devices));
    // double start_iterations, end_iterations;
    unsigned *lastGPU = NULL;

    //  int chosen[N];
    unsigned *occupancies = (unsigned *)calloc(ndevs, sizeof(*occupancies));
    unsigned long *gpuLoad = (unsigned long *)calloc(ndevs, sizeof(*gpuLoad));
    
    int timestep = 1;
    // int probSize = MAXWORK;
    int numThreads = 64;
    int numThreadsPerBlock = 256;
    // numThreads = omp_get_num_threads();
    int M = PSIZE, N = PSIZE, K = PSIZE;
    int check_result = 0;

    // srand((unsigned)time(NULL));
    float granularity = 0.9;
    if (argc <= 1)
    {
        printf("Usage bench_works [m] [n] [k] [granularity]\n");
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
        {
            granularity = atof(argv[4]);
            if (granularity < 0.0 || granularity > 1.0)
            {
                fprintf(stderr, "Error: granularity must be between 0.0 and 1.0. Received %lf.\n", granularity);
                exit(1); // Exit with error code 1
            }
        }
        if (argc > 5)
            numThreadsPerBlock = atoi(argv[5]);
        if (argc > 6)
            numThreads = atoi(argv[6]);
        if (argc > 7)
            check_result = 1;
    }
    int a_size = M * K, b_size = K * N, c_size = M * N;

    int rowsPerTask = MAX(1, (1.0 - granularity) * M);
    int numTasks = CEIL(M,rowsPerTask);
    // int streams_per_gpu = CEIL(numTasks,ndevs);
    int streams_per_gpu = 32;
    numThreadsPerBlock = CEIL(1024,streams_per_gpu);
    printf("bench_works [m=%d] [n=%d] [k=%d] [numTasks=%d] [granularity=%0.2lf] [rowsPerTask=%d] [numThreads=%d] [numThreadsPerBlock=%d] [resMatSize=%0.2e] [streams_per_gpu=%d]\n",
            M, N, K, numTasks, granularity, rowsPerTask, numThreads, numThreadsPerBlock, 1.0f*c_size, streams_per_gpu);

    #if defined(SCHED_ROUNDROBIN)
    printf("gpu_scheduler_static_rr,\t");
    #elif defined(SCHED_ADAPTIVE)
    printf("gpu_scheduler_dynamic_ad,\t");
    #elif defined(SCHED_ADAPTIVE2)
    printf("gpu_scheduler_dynamic_ad2,\t");
    #elif defined(SCHED_RANDOM)
    printf("gpu_scheduler_dynamic_random,\t");
    #elif defined(SCHED_DYNAMIC)
    printf("gpu_scheduler_dynamic_occ,\t");
    #elif defined(SCHED_DYNAMIC2)
    printf("gpu_scheduler_dynamic_occ2,\t");
    #else
    printf("none 0\n");
    #endif

    #if defined(ASYN)
    printf("asyn nowait\n");
    #else
    printf("syn with wait\n");
    #endif

    float *a,*b,*c;

    checkCuda(cudaMallocHost(&a,a_size*sizeof(float)));
    checkCuda(cudaMallocHost(&b,b_size*sizeof(float)));
    checkCuda(cudaMallocHost(&c,c_size*sizeof(float)));

    int *taskWork = (int *)malloc(sizeof(int) * numTasks);

    int *chosen = (int *)malloc(sizeof(int) * numTasks);
    int *success = (int *)malloc(sizeof(int) * numTasks);

    // initialize

    for (int i = 0; i < a_size; i++)
        // a[i] = (float)rand() / RAND_MAX * 2.0 - 1.0;
        a[i] = i%4;

    for (int i = 0; i < b_size; i++)
        // b[i] = (float)rand() / RAND_MAX * 2.0 - 1.0;
        b[i] = 4+i%3;

    for (int i = 0; i < c_size; i++)
        c[i] = 0.0;

    // printMatrix(a,M,K);printMatrix(b,K,N);printMatrix(c,M,N);
    std::vector<std::vector<cudaStream_t>> streams(ndevs,std::vector<cudaStream_t>(streams_per_gpu));
    for(int d=0;d<ndevs;d++){
        cudaSetDevice(d);
        for(int s=0;s<streams_per_gpu;s++)
            cudaStreamCreate(&streams[d][s]);
    }
    std::vector<int> strm_ctr(ndevs,0);

    auto nxt_strm = [&](int& x) -> int {
        int temp;
    #if defined(USEOPENMP)
        #pragma omp critical
        {
    #endif
    #if not defined(USEOPENMP)
            std::lock_guard<std::mutex> lock(mtx);
    #endif
            temp = x;
            x = (x+1)%streams_per_gpu;
    #if defined(USEOPENMP)
        }
    #endif
        return temp;
    };

    std::vector<int> startIndexes = generateEqualChunkStartIndices(M, numTasks);;
    
    // startIndexes = generateUniformChunkStartIndices(M, numTasks);
    startIndexes = generateRandomChunkStartIndices(M, numTasks);

    // std::cout << "Starting indices of chunks: ";
    // for (int index : startIndexes) {
    //     std::cout << index << " ";
    // }
    // std::cout << std::endl;
    
    // Calculate chunk sizes from start indices
    std::vector<int> chunkSizes = calculateChunkSizes(startIndexes, M);

    calculateStandardDeviation(chunkSizes, calculateMean(chunkSizes));

    std::vector<float*> d_b_global(ndevs);

    std::vector<std::thread> threads;

    transposeMatrix(b,K,N);
    
    #if defined(VECTORIZE)
    printf("vectorized,\t");
    #else
    printf("non-vectorized,\t");
    #endif

    #if defined(USEOPENMP)
    printf("openMP,\t");
    #else
    printf("non-openMP,\t");
    #endif


    start_timer();

    #if defined(PRE_TRANSFER)
    printf("PRE TRANSFER\n");
    #if defined(USEOPENMP)
    #pragma omp parallel for schedule(static,1)
    #endif
    for(int d=0;d<ndevs;d++){
        #if not defined(USEOPENMP)
        threads.push_back(std::thread([&, d]() {
        #endif
            cudaSetDevice(d);

            int nxt = nxt_strm(strm_ctr[d]);
            auto stream = streams[d][nxt];
            
            cudaMallocAsync(&d_b_global[d],b_size*sizeof(float),stream);
            cudaMemcpyAsync(d_b_global[d],b,b_size*sizeof(float),cudaMemcpyHostToDevice,stream);
            
            cudaDeviceSynchronize();
        
        #if not defined(USEOPENMP)
        }));
        #endif
    }

    joinThreads(threads);
    #else
    printf("No pre transfer\n");
    #endif
    int nextTask = ndevs;

    #if defined(USEOPENMP)
    #pragma omp parallel for schedule(static,1)
    #endif
    for (int i = 0; i < numTasks; i++){
        // printf("thread %d\ti %d\n",omp_get_thread_num(),i);
        #if not defined(USEOPENMP)
        threads.push_back(std::thread([&,i](){
        #endif
            // int start = i*rowsPerTask, end = MIN((i+1)*rowsPerTask,M);
            int start = startIndexes[i], end = (i==numTasks-1 ? M : startIndexes[i+1]);
            int nRows = end-start;
            float *d_a, *d_b, *d_c;
            int a_start, b_start, c_start, a_items, b_items, c_items, m, n, k;
            
            m=nRows; n=N; k=K;
            a_start = start*K; b_start = 0;   c_start = start*N;
            a_items = nRows*K; b_items = K*N; c_items = nRows*N;
            
            const int NNsq = c_items;

            #if defined(SCHED_ROUNDROBIN)
            const int dev = gpu_scheduler_static_rr(i, ndevs);
            #elif defined(SCHED_ADAPTIVE)
            const int dev = gpu_scheduler_dynamic_ad(gpuLoad, ndevs, NNsq);
            #elif defined(SCHED_ADAPTIVE2)
            const int dev = gpu_scheduler_dynamic_ad2(gpuLoad, ndevs, NNsq);
            #elif defined(SCHED_RANDOM)
            const int dev = gpu_scheduler_dynamic_random(occupancies, ndevs);
            #elif defined(SCHED_DYNAMIC)
            const int dev = gpu_scheduler_dynamic_occ(occupancies, ndevs);
            #elif defined(SCHED_DYNAMIC2)
            const int dev = gpu_scheduler_dynamic_occ2(occupancies, ndevs);
            #else
            const int dev = 0;
            #endif
            if (dev != -1)
                chosen[i] = dev;
            success[i] = 0;

            int d = chosen[i]; // assert(0 <= chosen[i] <= ndevs-1)

            devices[d]++;

            int nxt = nxt_strm(strm_ctr[d]);
            // printf("dev %d [%d] (%d,%d) GPU, stream: [%d, %d]\n",d,i,start,end,d,nxt);
            
            cudaSetDevice(d);
            auto stream = streams[d][nxt];

            cudaMallocAsync(&d_a,a_items*sizeof(float),stream);
            cudaMemcpyAsync(d_a,a+a_start,a_items*sizeof(float),cudaMemcpyHostToDevice,stream);
            
            #if not defined(PRE_TRANSFER)
            cudaMallocAsync(&d_b,b_items*sizeof(float),stream);
            cudaMemcpyAsync(d_b,b+b_start,b_items*sizeof(float),cudaMemcpyHostToDevice,stream);
            #endif
            
            cudaMallocAsync(&d_c,c_items*sizeof(float),stream);
            cudaMemcpyAsync(d_c,c+c_start,c_items*sizeof(float),cudaMemcpyHostToDevice,stream);
            
            
            int blk_x = min(MAX_TPB,n), blk_y = min(MAX_TPB,m);
            // dim3 blocksPerGrid(CEIL(n,blk_x), CEIL(m,blk_y));
            // dim3 threadsPerBlock(blk_x, blk_y); // Assuming width and height are within max threads per block limit 
            dim3 blocksPerGrid(CEIL(m*n,numThreadsPerBlock),1);
            dim3 threadsPerBlock(numThreadsPerBlock,1);
            // dim3 blocksPerGrid(1,1);
            // dim3 threadsPerBlock(1,1);
            // printf(" %d %d\n",blocksPerGrid.x,threadsPerBlock.x);
            
            #if defined(PRE_TRANSFER)
            multiply_kernel<<<blocksPerGrid,threadsPerBlock,0,stream>>>(d_a,d_b_global[d],d_c,start,m,n,k);
            #else
            multiply_kernel<<<blocksPerGrid,threadsPerBlock,0,stream>>>(d_a,d_b,d_c,start,m,n,k);
            #endif
            cudaMemcpyAsync(c+c_start,d_c,c_items*sizeof(float),cudaMemcpyDeviceToHost,stream);

            cudaFreeAsync(d_a,stream);
            #if not defined(PRE_TRANSFER)
            cudaFreeAsync(d_b,stream);
            #endif
            cudaFreeAsync(d_c,stream);
            
            // gpu_check<<<1,1,0,stream>>>(i,d);

            #if defined(SCHED_RANDOM) || defined(SCHED_DYNAMIC) || defined(SCHED_DYNAMIC2)
            success[i] = 1;
            cudaStreamSynchronize(stream);
            occupancies[d]--;
            #endif
            #if defined(SCHED_ADAPTIVE) || defined(SCHED_ADAPTIVE2)
            cudaStreamSynchronize(stream);
            success[i] = 1;
            gpuLoad[d] -= NNsq;
            // nextTask assignedTo the GPU just freed                                                                                                                                                                      
            int myTask;
            #pragma omp atomic capture 
            myTask = nextTask++;
            if(myTask < numTasks) chosen[myTask] = d;
            #endif
            // printf("dev %d [%d] (%d,%d) GPU, stream: [%d, %d]\n",d,i,start,end,d,nxt);
        
        #if not defined(USEOPENMP)
        }));
        #endif
    }
    
    joinThreads(threads);

    #if defined(USEOPENMP)
    #pragma omp parallel for schedule(static,1)
    #endif
    for(int d=0;d<ndevs;d++){
        #if not defined(USEOPENMP)
        threads.push_back(std::thread([&, d]()
        {
        #endif
            cudaSetDevice (d);
            cudaDeviceSynchronize();
        #if not defined(USEOPENMP)
        }));
        #endif
    }

    joinThreads(threads);

    #if defined(PRE_TRANSFER)
    #if defined(USEOPENMP)
    #pragma omp parallel for schedule(static,1)
    #endif
    for(int d=0;d<ndevs;d++){
        #if not defined(USEOPENMP)
        threads.push_back(std::thread([&, d]() {
        #endif
            cudaSetDevice (d);
            int nxt = nxt_strm(strm_ctr[d]);
            auto stream = streams[d][nxt];
            cudaFreeAsync(d_b_global[d],stream);
            cudaDeviceSynchronize();
        #if not defined(USEOPENMP)
        }));
        #endif
    }
    #endif
    
    joinThreads(threads);

    end_timer("GPU multiplication");
    transposeMatrix(b,K,N);

    std::vector<int> percent(ndevs,0);
    for(int i=0;i<numTasks;i++) percent[chosen[i]]++;
    for(int i=0;i<ndevs;i++) printf("GPU %d: %0.2lf  ",i,(double)percent[i]/numTasks);
    printf("\n"); 

    if(check_result){
        float *c_cpu;
        // checkCuda(cudaMallocHost(&c_cpu,c_size*sizeof(float)));
        c_cpu = (float*)malloc(c_size*sizeof(float));
        start_timer();
        printf("GPU Done... now checking correctness\n");
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

    for(auto &dev: streams)
        for(auto &str: dev)
            cudaStreamDestroy(str);

    printf("DONE\n\n");
}