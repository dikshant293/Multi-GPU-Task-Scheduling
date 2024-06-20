#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <assert.h>
#include <thread>
#include <vector>
#include <iostream>
#include <atomic>
#include <mutex>
#include <string>

#define MAX(x, y) (((x) > (y)) ? (x) : (y))
#define MIN(x, y) (((x) < (y)) ? (x) : (y))
#define ABS(x) (((x) < (0)) ? (-x) : (x))

#define BLOCK_SIZE 4
#define MAX_TPB 32

#define MM
#define PSIZE 20

#define EPSILON 1e-4

#define SCHED_ROUNDROBIN

// using data_type = float;

std::mutex mtx;

cudaError_t checkCuda(cudaError_t status)
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

// Kernel for matrix-matrix multiplication
__global__ void multiply_kernel(float *A, float *B, float *C, int rowStart, int M, int N, int K)
{
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    i += rowStart;
    if(i < M and j < N)
    {
        float sum = 0.0;
        for (int k = 0; k < K; ++k)
        {
            sum += A[i * K + k] * B[k * N + j];
        }
        C[i * N + j] = sum;
    }
}

void cudaMallocCheck(cudaError_t status) {
    if (status != cudaSuccess) {
        fprintf(stderr, "CUDA error: cudaMalloc failed with error %s\n", cudaGetErrorString(status));
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

    // assert(ndevs > 0);
    printf("There are %d GPUs\n", ndevs);
    int *devices = (int *)calloc(ndevs, sizeof(*devices));
    // double start_iterations, end_iterations;
    unsigned *lastGPU = NULL;

    //  int chosen[N];
    unsigned *occupancies = (unsigned *)calloc(ndevs, sizeof(*occupancies));
    unsigned long *gpuLoad = (unsigned long *)calloc(ndevs, sizeof(*gpuLoad));

    int timestep = 1;
    // int probSize = MAXWORK;
    int numThreads = 64;
    // numThreads = omp_get_num_threads();
    int M = PSIZE, N = PSIZE, K = PSIZE;
    int check_result = 0;

    srand((unsigned)time(NULL));
    float granularity = 0.5;
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
            numThreads = atoi(argv[5]);
        if (argc > 6)
            check_result = 1;
    }
    int a_size = M * K, b_size = K * N, c_size = M * N;

    int rowsPerTask = MAX(1, (1.0 - granularity) * M);
    int numTasks = (M + rowsPerTask - 1) / rowsPerTask;
    printf("bench_works [m=%d] [n=%d] [k=%d] [numTasks=%d] [granularity=%0.2lf] [rowsPerTask=%d] [numThreads=%d] [resMatSize=%0.2e] \n", M, N, K, numTasks, granularity, rowsPerTask, numThreads, 1.0f*c_size);

    #if defined(SCHED_ROUNDROBIN)
    printf("gpu_scheduler_static_rr\n");
    #elif defined(SCHED_ADAPTIVE)
    printf("gpu_scheduler_dynamic_ad\n");
    #elif defined(SCHED_ADAPTIVE2)
    printf("gpu_scheduler_dynamic_ad2\n");
    #elif defined(SCHED_RANDOM)
    printf("gpu_scheduler_dynamic_random\n");
    #elif defined(SCHED_DYNAMIC)
    printf("gpu_scheduler_dynamic_occ\n");
    #elif defined(SCHED_DYNAMIC2)
    printf("gpu_scheduler_dynamic_occ2\n");
    #else
    printf("none 0\n");
    #endif

    #if defined(ASYN)
    printf("asyn nowait\n");
    #else
    printf("syn with wait\n");
    #endif

    float *a,*b,*c,*c_cpu;

    cudaMallocHost(&a,a_size*sizeof(float));
    cudaMallocHost(&b,b_size*sizeof(float));
    cudaMallocHost(&c,c_size*sizeof(float));
    cudaMallocHost(&c_cpu,c_size*sizeof(float));

    int *taskWork = (int *)malloc(sizeof(int) * numTasks);

    int *chosen = (int *)malloc(sizeof(int) * numTasks);
    int *success = (int *)malloc(sizeof(int) * numTasks);

    // initialize

    for (int i = 0; i < a_size; i++)
        a[i] = (float)rand() / RAND_MAX * 2.0 - 1.0;
        // a[i] = i;

    for (int i = 0; i < b_size; i++)
        b[i] = (float)rand() / RAND_MAX * 2.0 - 1.0;
        // b[i] = i+400;

    for (int i = 0; i < c_size; i++)
        c[i] = 0.0;

    for (int i = 0; i < c_size; i++)
        c_cpu[i] = 0.0;

    // printMatrix(a,M,K);printMatrix(b,K,N);printMatrix(c,M,N);printMatrix(c_cpu,M,N);
    int streams_per_gpu = 4;
    std::vector<std::vector<cudaStream_t>> streams(ndevs,std::vector<cudaStream_t>(streams_per_gpu));
    for(int d=0;d<ndevs;d++){
        cudaSetDevice(d);
        for(int s=0;s<streams_per_gpu;s++)
            cudaStreamCreate(&streams[d][s]);
    }
    std::vector<int> strm_ctr(ndevs,0);

    auto nxt_strm = [&](int& x) -> int {
        std::lock_guard<std::mutex> lock(mtx);
        int temp = x;
        x = (x+1)%streams_per_gpu;
        return temp;
    };

    float *(*dev_pointers)[3];
    dev_pointers = (float *(*)[3])malloc(ndevs * sizeof(*dev_pointers));

    std::vector<std::thread> threads;

    start_timer();

    for(int d=0;d<ndevs;d++){
        threads.push_back(std::thread([&, d]()
        {
            float **d_a = &dev_pointers[d][0];
            float **d_b = &dev_pointers[d][1];
            float **d_c = &dev_pointers[d][2];

            cudaSetDevice(d);
            cudaMalloc(d_a,a_size*sizeof(float));
            cudaMalloc(d_b,b_size*sizeof(float));
            cudaMalloc(d_c,c_size*sizeof(float));

            // cudaMemcpy(*d_a,a,a_size*sizeof(float),cudaMemcpyHostToDevice);
            // cudaMemcpy(*d_b,b,b_size*sizeof(float),cudaMemcpyHostToDevice);
            // cudaMemcpy(*d_c,c,c_size*sizeof(float),cudaMemcpyHostToDevice);

            cudaMemcpyAsync(*d_a,a,a_size*sizeof(float),cudaMemcpyHostToDevice,streams[d][nxt_strm(strm_ctr[d])]);
            cudaMemcpyAsync(*d_b,b,b_size*sizeof(float),cudaMemcpyHostToDevice,streams[d][nxt_strm(strm_ctr[d])]);
            cudaMemcpyAsync(*d_c,c,c_size*sizeof(float),cudaMemcpyHostToDevice,streams[d][nxt_strm(strm_ctr[d])]);
            
            cudaDeviceSynchronize();
        }));
    }

    for (auto &thread: threads)
        thread.join();
    threads.clear();

    end_timer("host to device copy");

    // for(int d=0;d<ndevs;d++){
    //     float **d_a = &dev_pointers[d][0];
    //     float **d_b = &dev_pointers[d][1];
    //     float **d_c = &dev_pointers[d][2];
        
    //     cudaSetDevice(0);
    //     printMatrixGPU(*d_a,M,K);
    //     printMatrixGPU(*d_b,K,N);
    //     printMatrixGPU(*d_c,M,N);
    //     cudaDeviceSynchronize();
    // }
    start_timer();
    for (int i = 0; i < numTasks; i++)
    {
        threads.push_back(std::thread([&,i](){

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

            int start = i*rowsPerTask, end = MIN((i+1)*rowsPerTask,M);
            int nRows = end-start;

            int d = chosen[i]; // assert(0 <= chosen[i] <= ndevs-1)
            printf("dev %d [%d] (%d,%d)\n",d,i,start,end);

            float **d_a = &dev_pointers[d][0];
            float **d_b = &dev_pointers[d][1];
            float **d_c = &dev_pointers[d][2];

            devices[d]++;

            // Launch kernel for mative GPU matrix multiplication
            dim3 blocksPerGrid((N+min(MAX_TPB,N)-1)/min(MAX_TPB,N), (nRows+min(MAX_TPB,nRows)-1)/min(MAX_TPB,nRows));
            dim3 threadsPerBlock(min(MAX_TPB,N), min(MAX_TPB,nRows)); // Assuming width and height are within max threads per block limit 
            cudaSetDevice (d);
            multiply_kernel<<<blocksPerGrid,threadsPerBlock,0,streams[d][nxt_strm(strm_ctr[d])]>>>(*d_a,*d_b,*d_c,start,M,N,K);
            // cudaDeviceSynchronize();
            success[i] = 1;

        }));
    }

    for (auto &thread: threads)
        thread.join();
    threads.clear();
    
    for(int d=0;d<ndevs;d++)
        cudaDeviceSynchronize();

    end_timer("multiplication");

    start_timer();
    
    for(int d=0;d<ndevs;d++){
            float **d_c = &dev_pointers[d][2];
            cudaMemcpy(c_cpu,*d_c,c_size*sizeof(float),cudaMemcpyDeviceToHost);
            cudaDeviceSynchronize();
            int chk_size = (c_size+numThreads-1)/numThreads;
            for(int i=0;i<numThreads;i++){
                threads.push_back(std::thread([&, i]()
                {
                    // printf("i=%d (%d,%d)\n",i,i*chk_size,MIN(c_size,(i+1)*chk_size));
                    for(int j=i*chk_size; j<MIN(c_size,(i+1)*chk_size); j++) c[j]+=c_cpu[j];
                }));
            }
            for (auto &thread: threads)
                thread.join();
            threads.clear();
    }
    

    end_timer("device to host");

    if(check_result){
        printf("GPU Done... now checking correctness\n");
        for (int ii = 0; ii < M; ii++)
            for (int jj = 0; jj < N; jj++){
                c_cpu[ii * N + jj] = 0.0;
                for (int kk = 0; kk < K; kk++)
                    c_cpu[ii * N + jj] += a[ii * K + kk] * b[kk * N + jj];
            }

        // printMatrix(c,M,N);
        // printMatrix(c_cpu,M,N);

        bool flag = true;
        for (int i = 0; i < M; i++)
        {
            for (int j = 0; j < N; j++){
                float x = c[i * N + j], y = c_cpu[i * N + j];
                if (x != y && ABS((x - y)) > EPSILON) // data_type precision comparision upto 10^-6 for types like doubles
                {
                    printf("(%d,%d) : got %lf expected %lf diff %e\n",i,j,x,y,ABS(x - y));
                    flag = false;
                    break;
                }
            }
            if (!flag)
                break;
        }
        printf("Correctness check: %s\n",(flag ? "PASSED" : "FAILED"));
    }
    // printf("Sleeping\n");
    // std::this_thread::sleep_for(std::chrono::seconds(5));
    
    for(int d=0;d<ndevs;d++){
        threads.push_back(std::thread([&, d]()
        {
            float **d_a = &dev_pointers[d][0];
            float **d_b = &dev_pointers[d][1];
            float **d_c = &dev_pointers[d][2];

            cudaSetDevice(d);
            cudaFreeAsync(d_a,streams[d][nxt_strm(strm_ctr[d])]);
            cudaFreeAsync(d_b,streams[d][nxt_strm(strm_ctr[d])]);
            cudaFreeAsync(d_c,streams[d][nxt_strm(strm_ctr[d])]);

            cudaDeviceSynchronize();
        }));
    }

    for(auto &thread: threads)
        thread.join();
    threads.clear();
    
    cudaFreeHost(a);
    cudaFreeHost(b);
    cudaFreeHost(c);
    cudaFreeHost(c_cpu);

    printf("DONE\n");
}