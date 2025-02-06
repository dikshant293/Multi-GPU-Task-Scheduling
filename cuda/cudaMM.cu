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
#include <random>

#define CEIL(x, y) (((x) + (y) - 1) / (y))
#define MAX(x, y) (((x) > (y)) ? (x) : (y))
#define MIN(x, y) (((x) < (y)) ? (x) : (y))
#define ABS(x) (((x) < (0)) ? (-x) : (x))

#define BLOCK_SIZE 4
#define MAX_TPB 32

#define MM
#define PSIZE 2000

#define EPSILON 1e-6


std::mutex mtx;

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

__host__ inline unsigned gpu_scheduler_dynamic_occ2(unsigned *occupancies, int ngpus, int taskID)
{
    int chosen = -1;
    while (chosen == -1)
    {
        for (unsigned i = 0; i < ngpus; i++)
        {
#pragma omp critical
            {
                int g = (taskID + i)%ngpus;
                if (occupancies[g] == 0)
                {
                    occupancies[g]++;
                    chosen = g;
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

__host__ inline unsigned gpu_scheduler_mem(int ngpus, int task_id)
{
    short looking = 1;
    unsigned chosen;
    while (looking)
    {
        size_t max_free = 0;
    #pragma omp critical
        {
            for (unsigned j = 0; j < ngpus; j++)
            {
                size_t free_byte;
                size_t total_byte;
                int i = (j + task_id)%ngpus;
                cudaSetDevice(i);
                cudaError_t cuda_status = cudaMemGetInfo(&free_byte, &total_byte);

                if (cudaSuccess != cuda_status) {
                    std::cerr << "Error: cudaMemGetInfo fails, " << cudaGetErrorString(cuda_status) << std::endl;
                    exit(1);
                }
                if (free_byte > max_free)
                {
                    max_free = free_byte;
                    chosen = i;
                }
            }
        }
        looking = 0;
        break;
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
    cudaDeviceSynchronize();
}

// global clock timer
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

std::vector<int> generateChunkStartIndicesWithSD(int n, int m, double sd) {
    std::vector<int> startIndexes;
    std::vector<int> chunkSizes(m);
    int sumOfSizes = 0;
    
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<> dist(n / m, sd);  // Normal distribution centered around the base size

    // Generate initial chunk sizes
    for (int i = 0; i < m; ++i) {
        int size = std::round(dist(gen));  // Generate and round to nearest integer
        chunkSizes[i] = size > 0 ? size : 0;  // Ensure no negative sizes
        sumOfSizes += chunkSizes[i];
    }

    // Adjust the chunk sizes to exactly sum up to 'n'
    int error = sumOfSizes - n;
    while (error != 0) {
        for (int i = 0; i < m && error != 0; ++i) {
            if (error > 0 && chunkSizes[i] > 0) {  // Need to reduce the sum
                chunkSizes[i]--;
                error--;
            } else if (error < 0) {  // Need to increase the sum
                chunkSizes[i]++;
                error++;
            }
        }
    }

    // Generate starting indices from adjusted chunk sizes
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
    unsigned *lastGPU = NULL;

    unsigned *occupancies = (unsigned *)calloc(ndevs, sizeof(*occupancies));
    unsigned long *gpuLoad = (unsigned long *)calloc(ndevs, sizeof(*gpuLoad));
    
    int timestep = 1;
    int numThreads = 64;
    int numThreadsPerBlock = 1024;
    int M = PSIZE, N = PSIZE, K = PSIZE;
    int check_result = 0;
    int streams_per_gpu = 4;
    float granularity = 0.9;
    int chunk = 0;
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
            chunk = atoi(argv[5]);
        if (argc > 6)
            check_result = atoi(argv[6]);
        if (argc > 7)
            streams_per_gpu = atoi(argv[7]);
    }
    int a_size = M * K, b_size = K * N, c_size = M * N;

    int rowsPerTask = MAX(1, (1.0 - granularity) * M);
    int numTasks = CEIL(M,rowsPerTask);
    printf("bench_works [m=%d] [n=%d] [k=%d] [numTasks=%d] [granularity=%lf] [numThreadsPerBlock=%d] [streams_per_gpu=%d]\n",
            M, N, K, numTasks, granularity, numThreadsPerBlock, streams_per_gpu);

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
    #elif defined(SCHED_MEM)
    printf("gpu_scheduler_mem,\t");
    #else
    printf("none 0\n");
    #endif
    
    float *a,*b,*c;

    checkCuda(cudaMallocHost(&a,a_size*sizeof(float)));
    checkCuda(cudaMallocHost(&b,b_size*sizeof(float)));
    checkCuda(cudaMallocHost(&c,c_size*sizeof(float)));

    int *taskWork = (int *)malloc(sizeof(int) * numTasks);
    int *chosen = (int *)malloc(sizeof(int) * numTasks);
    int *success = (int *)malloc(sizeof(int) * numTasks);

    // initialize matrices
    for (int i = 0; i < a_size; i++)
        a[i] = i%4; // can be replace with any random value

    for (int i = 0; i < b_size; i++)
        b[i] = 4+i%3; // can be replace with any random value

    for (int i = 0; i < c_size; i++)
        c[i] = 0.0; // can be replace with any random value

    std::vector<int> startIndexes = generateEqualChunkStartIndices(M, numTasks);;
    
    if(chunk==1) startIndexes = generateUniformChunkStartIndices(M, numTasks);
    if(chunk==2) startIndexes = generateRandomChunkStartIndices(M, numTasks);
    if(chunk==3){
        float sd = 10.0;
        if(argc>8) sd = atof(argv[8]);
        startIndexes = generateChunkStartIndicesWithSD(M, numTasks, sd);
    }

    // print standard deviation and mean of the chunk sizes
    std::vector<int> chunkSizes = calculateChunkSizes(startIndexes, M);
    printf("Task Sizes: ");
    calculateStandardDeviation(chunkSizes, calculateMean(chunkSizes));

    std::vector<float*> d_a_global(ndevs), d_b_global(ndevs), d_c_global(ndevs);
    std::vector<std::thread> threads;

    // print what flags are set and not set
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
    
    #if defined(USE1D)
    printf("1d blocks\t");
    #else
    printf("2d blocks\t");
    #endif
    
    // defining and initializing the streams for each GPU
    std::vector<std::vector<cudaStream_t>> streams(ndevs,std::vector<cudaStream_t>(streams_per_gpu));
    #pragma omp parallel for schedule(static,1)
    for(int d=0;d<ndevs;d++){
        cudaSetDevice(d);
        
        cudaMemPool_t mempool;
        cudaDeviceGetDefaultMemPool(&mempool, d);
        uint64_t threshold = UINT64_MAX;
        cudaMemPoolSetAttribute(mempool, cudaMemPoolAttrReleaseThreshold, &threshold);
        
        for(int s=0;s<streams_per_gpu;s++)
            cudaStreamCreate(&streams[d][s]);
        cudaDeviceSynchronize();
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
    
    start_timer();
    // transpose B matrix for better cache locality
    transposeMatrix(b,K,N);

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

    // start CPU threads for task scheduling and distribution
    #if defined(USEOPENMP)
    #pragma omp parallel for schedule(static,1)
    #endif
    for (int i = 0; i < numTasks; i++){
        #if not defined(USEOPENMP)
        threads.push_back(std::thread([&,i](){
        #endif
            int start = startIndexes[i], end = (i==numTasks-1 ? M : startIndexes[i+1]);
            int nRows = end-start;
            float *d_a, *d_b, *d_c;
            int a_start, b_start, c_start, a_items, b_items, c_items, m, n, k;
            
            m=nRows; n=N; k=K;
            a_start = start*K; b_start = 0;   c_start = start*N;
            a_items = nRows*K; b_items = K*N; c_items = nRows*N;
            
            const int NNsq = c_items;

            // get target GPU from scheduling strategy
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
            const int dev = gpu_scheduler_dynamic_occ2(occupancies, ndevs, i);
            #elif defined(SCHED_MEM)
            const int dev = gpu_scheduler_mem(ndevs,i);
            #else
            const int dev = 0;
            #endif
            if (dev != -1)
                chosen[i] = dev;
            success[i] = 0;

            int d = chosen[i]; // assert(0 <= chosen[i] <= ndevs-1)

            devices[d]++;
            int nxt = nxt_strm(strm_ctr[d]);
            
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
            
            // define block and threads per block
            int blk_x = min(MAX_TPB,n), blk_y = min(MAX_TPB,m);
            #if defined(USE1D)
            dim3 blocksPerGrid(CEIL(m*n,numThreadsPerBlock),1);
            dim3 threadsPerBlock(numThreadsPerBlock,1);
            #else
            dim3 blocksPerGrid(CEIL(n,blk_x), CEIL(m,blk_y));
            dim3 threadsPerBlock(blk_x, blk_y); // Assuming width and height are within max threads per block limit 
            #endif
            
            // launch kernel
            #if defined(PRE_TRANSFER)
            multiply_kernel<<<blocksPerGrid,threadsPerBlock,0,stream>>>(d_a,d_b_global[d],d_c,m,n,k);
            #else
            multiply_kernel<<<blocksPerGrid,threadsPerBlock,0,stream>>>(d_a,d_b,d_c,m,n,k);
            #endif
            cudaMemcpyAsync(c+c_start,d_c,c_items*sizeof(float),cudaMemcpyDeviceToHost,stream);

            // free task allocated memory
            cudaFreeAsync(d_a,stream);
            #if not defined(PRE_TRANSFER)
            cudaFreeAsync(d_b,stream);
            #endif
            cudaFreeAsync(d_c,stream);

            #if defined(SCHED_RANDOM) || defined(SCHED_DYNAMIC) || defined(SCHED_DYNAMIC2)
            success[i] = 1;
            cudaStreamSynchronize(stream);
            #pragma omp critical
            {
                occupancies[d]--;
            }
            #endif
            #if defined(SCHED_ADAPTIVE) || defined(SCHED_ADAPTIVE2)
            cudaStreamSynchronize(stream);
            success[i] = 1;
            #pragma omp critical
	    {
            gpuLoad[d] -= NNsq;
	    }
            // nextTask assignedTo the GPU just freed                                                                                                                                                                      
            int myTask;
            #pragma omp atomic capture 
            myTask = nextTask++;
            if(myTask < numTasks) chosen[myTask] = d;
            #endif
        
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

    // free pre-transferred memory (if any)
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

    transposeMatrix(b,N,K);
    end_timer("GPU multiplication");

    // print % distribution of tasks across GPUs
    std::vector<int> percent(ndevs,0);
    for(int i=0;i<numTasks;i++) percent[chosen[i]]++;
    for(int i=0;i<ndevs;i++) printf("GPU %d: %0.2lf  ",i,(double)percent[i]/numTasks);
    printf("\n"); 

    // check result correctness
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
        cudaFreeHost(c_cpu);
    }
    
    cudaFreeHost(a);
    cudaFreeHost(b);
    cudaFreeHost(c);

    for(auto &dev: streams)
        for(auto &str: dev)
            cudaStreamDestroy(str);

    printf("\n");
}
