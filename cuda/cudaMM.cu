#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>
#include <time.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <assert.h>
#include <thread>
#include <vector>

#define IMAGE_MAGIC_NUMBER 2051
#define LABEL_MAGIC_NUMBER 2049
#define IMAGE_WIDTH 28
#define IMAGE_HEIGHT 28
#define NUM_IMAGES 60000
#define NUM_VALIDATION_IMAGES 10000
#define NUM_TRAIN_IMAGES (NUM_IMAGES - NUM_VALIDATION_IMAGES)
#define NUM_TEST_IMAGES 10000
#define NUM_DIGITS 10

#define MAX(x, y) (((x) > (y)) ? (x) : (y))
#define MIN(x, y) (((x) < (y)) ? (x) : (y))
#define ABS(x) (((x) < (0)) ? (-x) : (x))

#define BLOCK_SIZE 16

#define MM
#define PSIZE 20
#define MAXWORK 10
#define MAX_LOOP 10

// using data_type = float;


inline unsigned gpu_scheduler_static_rr(int taskID, int ngpus)
{
    return taskID % ngpus;
}

inline unsigned gpu_scheduler_dynamic_ad(unsigned long *gpuLoad, int ngpus, int taskWeight)
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
inline unsigned gpu_scheduler_dynamic_ad2(unsigned long *gpuLoad, int ngpus, int taskWeight)
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

inline unsigned gpu_scheduler_dynamic_random(unsigned *occupancies, int ngpus)
{
    const unsigned chosen = rand() % ngpus;
#pragma omp atomic
    occupancies[chosen]++;
    return chosen;
}

inline unsigned gpu_scheduler_dynamic_occ2(unsigned *occupancies, int ngpus)
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

inline unsigned gpu_scheduler_dynamic_occ(unsigned *occupancies, int ngpus)
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


// Allocate memory for matrices A, B, and C on device
// float *d_A, *d_B, *d_C;

// Kernel for matrix-matrix multiplication
__global__ void multiply_kernel(float *A, float *B, float *C, int rowStart, int M, int N, int K)
{
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    i += rowStart;
    float a, b;
    if (i < M && j < N)
    {
        float sum = 0.0;
        for (int k = 0; k < K; ++k)
        {
            a = A[i * K + k];
            b = B[k * N + j];
            sum += a * b;
        }
        C[i * N + j] = sum;
    }
}

// void gpu_multiply_matrices(float *h_A, int transA, float *h_B, int transB, float *h_C, int m, int n, int k)
// {
//     // Copy matrices A and B from host to device
//     cudaMemcpy(d_A, h_A, m * k * sizeof(float), cudaMemcpyHostToDevice);
//     cudaMemcpy(d_B, h_B, k * n * sizeof(float), cudaMemcpyHostToDevice);

//     // Launch kernel for mative GPU matrix multiplication
//     dim3 threadsPerBlock(BLOCK_SIZE, BLOCK_SIZE);
//     dim3 numBlocks((n + threadsPerBlock.x - 1) / threadsPerBlock.x,
//                    (m + threadsPerBlock.y - 1) / threadsPerBlock.y);

//     multiply_kernel<<<numBlocks, threadsPerBlock>>>(d_A, transA, d_B, transB, d_C, m, n, k);
//     cudaMemcpy(h_C, d_C, m * n * sizeof(float), cudaMemcpyDeviceToHost);
// }

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

__global__ void printMatrixGPU(float *mat, int m, int n){
    for(int i=0;i<m;i++){
        for(int j=0;j<n;j++){
            printf("%0.2lf ",mat[i*n+j]);
        }
        printf("\n");
    }
    printf("\n");
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
    double start_iterations, end_iterations;
    unsigned *lastGPU = NULL;

    //  int chosen[N];
    unsigned *occupancies = (unsigned *)calloc(ndevs, sizeof(*occupancies));
    unsigned long *gpuLoad = (unsigned long *)calloc(ndevs, sizeof(*gpuLoad));

    int timestep = 1;
    // int probSize = MAXWORK;
    int numThreads = 1;
#pragma omp parallel
    numThreads = omp_get_num_threads();
    float granularity = 0.5;
    int M = PSIZE, N = PSIZE, K = PSIZE;
    // int numloop = MAX_LOOP;

    srand((unsigned)time(NULL));
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
        // if (argc > 4)
        //     numloop = atoi(argv[4]);
    }
    int a_size = M * K, b_size = K * N, c_size = M * N;

    int rowsPerTask = MAX(1, (1.0 - granularity) * M);
    int numTasks = (M + rowsPerTask - 1) / rowsPerTask;
    printf("bench_works [m=%d] [n=%d] [k=%d] [numTasks=%d] [granularity=%0.2lf] [rowsPerTask=%d] [numThreads=%d] \n", M, N, K, numTasks, granularity, rowsPerTask, numThreads);


    // cudaSetDevice(2);
    // float *x,*y,*d_x;
    // cudaMallocHost(&x,c_size*sizeof(float));
    // cudaMallocHost(&y,c_size*sizeof(float));
    // for (int i = 0; i < c_size; i++){
    //     x[i] = 6.9;
    //     y[i] = 0.0;
    // }
    // printMatrix(x,M,N);
    // printMatrix(y,M,N);
    // cudaMalloc((void **)&d_x, c_size * sizeof(float));
    // cudaMemcpy(d_x, x, c_size * sizeof(float), cudaMemcpyHostToDevice);
    // cudaDeviceSynchronize();
    // cudaMemcpy(y, d_x, c_size * sizeof(float), cudaMemcpyDeviceToHost);
    // cudaDeviceSynchronize();
    // printMatrix(y,M,N);

    // printf("AAAAAAAAAAAA\n");

    float *a,*b,*c,*c_cpu;

    // a = (float *)malloc(a_size * sizeof(float));
    // b = (float *)malloc(b_size * sizeof(float));
    // c = (float *)malloc(c_size * sizeof(float));
    // c_cpu = (float *)malloc(c_size * sizeof(float));

    cudaMallocHost(&a,a_size*sizeof(float));
    cudaMallocHost(&b,b_size*sizeof(float));
    cudaMallocHost(&c,c_size*sizeof(float));
    cudaMallocHost(&c_cpu,c_size*sizeof(float));

    int *taskWork = (int *)malloc(sizeof(int) * numTasks);

    int *chosen = (int *)malloc(sizeof(int) * numTasks);
    int *success = (int *)malloc(sizeof(int) * numTasks);

    // initialize

    for (int i = 0; i < a_size; i++)
        // a[i] = (float)rand() / RAND_MAX * 2.0 - 1.0;
        a[i] = 1.0;

    for (int i = 0; i < b_size; i++)
        // b[i] = (float)rand() / RAND_MAX * 2.0 - 1.0;
        b[i] = 2.0;

    for (int i = 0; i < c_size; i++)
        c[i] = -i;

    for (int i = 0; i < c_size; i++)
        c_cpu[i] = i;

    printMatrix(a,M,K);printMatrix(b,K,N);printMatrix(c,M,N);printMatrix(c_cpu,M,N);

    double cpu_time = 0.0;
    double task_time = 0.0;
    int nextTask = ndevs; // this is needed only for the ad2 and ad strategy

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

    float *(*dev_pointers)[3];
    dev_pointers = (float *(*)[3])malloc(ndevs * sizeof(*dev_pointers));

    std::vector<std::thread> threads;

    for (unsigned int d = 0; d < ndevs; d++)
    {
        // threads.push_back(std::thread([&, d]()
        // {
            cudaSetDevice(d);

            printf("GPU %d\n",d);
            float *d_a = dev_pointers[d][0];
            float *d_b = dev_pointers[d][1];
            float *d_c = dev_pointers[d][2];

            cudaMallocCheck(cudaMalloc((void **)&d_a, a_size * sizeof(float)));
            cudaMallocCheck(cudaMalloc((void **)&d_b, b_size * sizeof(float)));
            cudaMallocCheck(cudaMalloc((void **)&d_c, c_size * sizeof(float)));

            cudaDeviceSynchronize();

            // copy each matrix to the GPU
            cudaMemcpy(d_a, a, a_size * sizeof(float), cudaMemcpyHostToDevice);
            cudaMemcpy(d_b, b, b_size * sizeof(float), cudaMemcpyHostToDevice);
            cudaMemcpy(d_c, c, c_size * sizeof(float), cudaMemcpyHostToDevice);
            
            cudaDeviceSynchronize();
        // }));
    }

    // for (int d = 0; d < ndevs; ++d) {
    //     cudaSetDevice(d);
    //     cudaDeviceSynchronize();

    //     float *d_a = dev_pointers[d][0];
    //     float *d_b = dev_pointers[d][1];
    //     float *d_c = dev_pointers[d][2];
    //     printf("GPU sync %d\n",d);
    //     printMatrixGPU<<<1,1>>>(d_a,M,K);
    //     cudaDeviceSynchronize();
    // }

    printf("joining\n");

    for (auto &thread: threads)
        thread.join();

    threads.clear();
    
    printf("done with mem to\n");

    // for (int d = 0; d < ndevs; ++d) {
    //     cudaSetDevice(d);

    //     float *d_a = dev_pointers[d][0];
    //     float *d_b = dev_pointers[d][1];
    //     float *d_c = dev_pointers[d][2];
    //     printf("GPU sync %d\n",d);
    //     printMatrixGPU<<<1,1>>>(d_a,M,K);
    //     cudaDeviceSynchronize();
    // }

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

            float *d_a = dev_pointers[d][0];
            float *d_b = dev_pointers[d][1];
            float *d_c = dev_pointers[d][2];

            devices[d]++;

            // Launch kernel for mative GPU matrix multiplication
            dim3 threadsPerBlock(BLOCK_SIZE, BLOCK_SIZE);
            dim3 numBlocks((N + threadsPerBlock.x - 1) / threadsPerBlock.x,
                        (nRows + threadsPerBlock.y - 1) / threadsPerBlock.y);

            cudaSetDevice (d);

            // multiply_kernel<<<numBlocks,threadsPerBlock>>>(d_a,d_b,d_c,start,M,N,K);
            
            success[i] = 1;

        }));
    }

    for (int i = 0; i < ndevs; ++i) {
        cudaSetDevice(i);
        cudaDeviceSynchronize();
    }

    for (auto &thread: threads)
        thread.join();

    threads.clear();

    printf("Copying back\n");

    for (int d = 0; d < ndevs; d++)
    {
        
        // threads.push_back(std::thread([&, d]()
        // {
            cudaSetDevice(d);

            printf("GPU %d\n",d);
            float *d_a = dev_pointers[d][0];
            float *d_b = dev_pointers[d][1];
            float *d_c = dev_pointers[d][2];

            // Asynchronously copy each matrix to the GPU
            cudaMemcpy(c_cpu, d_a, a_size * sizeof(float), cudaMemcpyDeviceToHost);
        // }));
        
        cudaDeviceSynchronize();
        printMatrix(c_cpu,M,N);
    }

    for (int i = 0; i < ndevs; ++i) {
        cudaSetDevice(i);
        cudaDeviceSynchronize();
    }

    for (auto &thread: threads)
        thread.join();

    cudaFreeHost(a);
    cudaFreeHost(b);
    cudaFreeHost(c);

    for (int d = 0; d < ndevs; ++d) {
        cudaSetDevice(d);
        cudaFree(dev_pointers[d][0]);
        cudaFree(dev_pointers[d][1]);
        cudaFree(dev_pointers[d][2]);
    }

    printf("DONE\n");
}