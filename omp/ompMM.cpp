#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <assert.h>
#include <math.h>
#include <omp.h>
#include <limits.h>
#include <vector>
#include <iostream>
#include <string>
#include <algorithm>
#include <atomic>
#include <mutex>
#include <thread>
#include <cstdlib>
#include <numeric>
#include <cmath>
#include <ctime>
#include <vector_types.h>

// Input data distribution
#define RANDOM_SIZED_TASKS
// #define INCREASING_SIZED_TASKS
#define LOWERLT 128

// Application problem
#define MM
#define PSIZE 20
#define MAXWORK 10
#define MAX_LOOP 10

// Scheduling strategies, unset all to use the compact schedue

// #define SCHED_ROUNDROBIN
// #define SCHED_DYNAMIC
// #define SCHED_DYNAMIC2
// #define SCHED_RANDOM
// #define SCHED_ADAPTIVE
// #define SCHED_ADAPTIVE2


#define CEIL(x, y) (((x) + (y) - 1) / (y))
#define MAX(x, y) (((x) > (y)) ? (x) : (y))
#define MIN(x, y) (((x) < (y)) ? (x) : (y))
#define ABS(x) (((x) < (0)) ? (-x) : (x))
#define EPSILON 1e-6

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

void printMatrix(float *mat, int m, int n){
    for(int i=0;i<m;i++){
        for(int j=0;j<n;j++){
            printf("%lf ",mat[i*n+j]);
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

void transposeMatrix(float* matrix, int m, int n) {
    for (int i = 0; i < m; ++i) {
        for (int j = i + 1; j < n; ++j) {
            std::swap(matrix[i * n + j], matrix[j * n + i]);
        }
    }
}


inline void multiply(float *d_a, float *d_b, float *d_c, int M, int N, int K, int d, int a_items=0, int b_items=0, int c_items=0)
{
    #if defined(PRE_TRANSFER)
    #pragma omp target teams distribute parallel for num_teams(CEIL(M*N,1024)) thread_limit(1024) schedule (static, 1) device(d) is_device_ptr(d_a,d_b,d_c)
    #else
    #pragma omp target teams distribute parallel for num_teams(CEIL(M*N,1024)) thread_limit(1024) schedule (static, 1) device(d) is_device_ptr(d_b) map(to:d_a[0:a_items]) map(tofrom:d_c[0:c_items])
    #endif
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
            sum += d_a[ii * K + kk] * d_b[kk * N + jj];
            // sum += d_a[ii * K + kk] * d_b[jj * K + kk];
        }
        #endif
        d_c[ii * N + jj] = sum;
    }
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


int main(int argc, char *argv[])
{
    const int ndevs = omp_get_num_devices();
    assert(ndevs > 0);
    int *devices = (int *)calloc(ndevs, sizeof(*devices));
    double start_iterations, end_iterations;
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
    int chunk = 0;
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
            chunk = atoi(argv[5]);
        if (argc > 6)
            check_result = 1;
    }
    int a_size = M * K, b_size = K * N, c_size = M * N;

    int rowsPerTask = MAX(1, (1.0 - granularity) * M);
    int numTasks = CEIL(M,rowsPerTask);
    // int streams_per_gpu = CEIL(numTasks,ndevs);
    int streams_per_gpu = 32;
    numThreadsPerBlock = CEIL(1024,streams_per_gpu);
    printf("bench_works [m=%d] [n=%d] [k=%d] [numTasks=%d] [granularity=%lf] [rowsPerTask=%d]\n",
            M, N, K, numTasks, granularity, rowsPerTask);

    float *a = (float *)malloc(a_size * sizeof(float));
    float *b = (float *)malloc(b_size * sizeof(float));
    float *c = (float *)malloc(c_size * sizeof(float));

    int *taskWork = (int *)malloc(sizeof(int) * numTasks);
    int *taskWorkSquared = (int *)malloc(sizeof(int) * numTasks);

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

    int ctaskwork;

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
    
    #if defined(PRE_TRANSFER)
    printf("pre-transfered All\t");
    #else
    printf("pre transfer B matrix\t");
    #endif
    #if defined(VECTORIZE)
    printf("vectorized\t");
    #else
    printf("non vectorized\t");   
    #endif

    std::vector<int> startIndexes = generateEqualChunkStartIndices(M, numTasks);;
    
    if(chunk==1) startIndexes = generateUniformChunkStartIndices(M, numTasks);
    if(chunk==2) startIndexes = generateRandomChunkStartIndices(M, numTasks);
    
    std::vector<int> chunkSizes = calculateChunkSizes(startIndexes, M);
    printf("Task Sizes: ");
    calculateStandardDeviation(chunkSizes, calculateMean(chunkSizes));

    int host_id = omp_get_initial_device();
    float * __restrict (*dev_pointers)[3];
    dev_pointers = (float *(*)[3])malloc(ndevs * sizeof(*dev_pointers));
    
    #if defined(VECTORIZE)
    transposeMatrix(b,K,N);
    #endif

    start_timer();
    #pragma omp parallel for shared(dev_pointers)
    for(int d=0;d<ndevs;d++){
        dev_pointers[d][1] = (float *)omp_target_alloc(b_size * sizeof(float), d);
        omp_target_memcpy(dev_pointers[d][1], b, b_size * sizeof(float), 0, 0, d, host_id);

        #if defined(PRE_TRANSFER) 
        dev_pointers[d][0] = (float *)omp_target_alloc(a_size * sizeof(float), d);
        omp_target_memcpy(dev_pointers[d][0], a, a_size * sizeof(float), 0, 0, d, host_id);
        dev_pointers[d][2] = (float *)omp_target_alloc(c_size * sizeof(float), d);
        omp_target_memcpy(dev_pointers[d][2], c, c_size * sizeof(float), 0, 0, d, host_id);
        #endif
    }

    #pragma omp parallel for shared(success, nextTask, chosen, startIndexes, chunkSizes) schedule(static,1)
    for (int i = 0; i < numTasks; i++)
    {
        int start = startIndexes[i], end = (i==numTasks-1 ? M : startIndexes[i+1]);
        int nRows = end-start;
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

        int d = chosen[i]; 
        float *d_a,*d_b,*d_c;
        d_b = dev_pointers[d][1];

        #if defined(PRE_TRANSFER)
        d_a = dev_pointers[d][0];
        d_c = dev_pointers[d][2];
        #endif
        devices[d]++;
        #if defined(PRE_TRANSFER)
        multiply(d_a+a_start,d_b+b_start,d_c+c_start,m,n,k,d);
        omp_target_memcpy(c+c_start, d_c+c_start, c_items * sizeof(float), 0, 0, host_id, d);
        #else
        multiply(a+a_start,d_b+b_start,c+c_start,m,n,k,d,a_items,b_items,c_items);
        #endif
        success[i] = 1;
        #if not defined(PRE_TRANSFER)
        // omp_target_free(d_a, d);
        // omp_target_free(d_c, d);
        #endif
            
        #if defined(SCHED_RANDOM) || defined(SCHED_DYNAMIC) || defined(SCHED_DYNAMIC2)
        occupancies[d]--;
        #endif
        #if defined(SCHED_ADAPTIVE) || defined(SCHED_ADAPTIVE2)
        gpuLoad[d] -= NNsq;
        // nextTask assignedTo the GPU just freed
        int myTask;
        #pragma omp atomic capture
        myTask = nextTask++;

        if (myTask < numTasks)
            chosen[myTask] = d;
        #endif
        // }
    } // end taskloop
        
    #pragma omp taskwait
    for(int d=0;d<ndevs;d++){
        omp_target_free(dev_pointers[d][1], d);
        #if defined(PRE_TRANSFER)
        omp_target_free(dev_pointers[d][0], d);
        omp_target_free(dev_pointers[d][2], d);
        #endif
    }

    int check = 0;
    end_timer("GPU Multiplication");

    std::vector<int> percent(ndevs,0);
    for(int i=0;i<numTasks;i++) percent[chosen[i]]++;
    for(int i=0;i<ndevs;i++) printf("GPU %d: %0.2lf  ",i,(double)percent[i]/numTasks);
    printf("\n");
    
    #if defined(VECTORIZE)
    transposeMatrix(b,N,K);
    #endif

    int lastFail = 0;
    for (int i = 0; i < numTasks; i++)
    {
        check += success[i];
        if (success[i] == 0)
            lastFail = i;
    }
    if (check != numTasks)
    {
        printf("failed! LastFailed %d \n", lastFail);
    }


    // printf("Total number of CPU threads=%d\n", omp_get_num_threads());
    if(check_result){
        printf("GPU Done... now checking correctness\n");

        float *c_cpu = (float *)malloc(c_size * sizeof(float));
        for (int i = 0; i < c_size; i++)
        c_cpu[i] = 0.0;
        start_timer();
        #pragma omp parallel for
        for (int ii = 0; ii < M; ii++)
            for (int jj = 0; jj < N; jj++)
                for (int kk = 0; kk < K; kk++)
                    c_cpu[ii * N + jj] += a[ii * K + kk] * b[kk * N + jj];
        end_timer("CPU Multiplication");
        bool flag = true;
        for (int i = 0; i < M; i++)
        {
            for (int j = 0; j < N; j++){
                float x = c[i * N + j], y = c_cpu[i * N + j];
                float diff = x-y;
                if (x != y && ABS(diff) > EPSILON) // float precision comparision upto 10^-6 for types like doubles
                {
                    printf("(%d,%d) : got %lf expected %lf diff %e\n",i,j,x,y,ABS(diff));
                    flag = false;
                    break;
                }
            }
            if (!flag)
                break;
        }
        printf("Correctness check: %s\n",(flag ? "PASSED" : "FAILED"));
        // printMatrix(a,M,K);printMatrix(b,K,N);printMatrix(c,M,N);printMatrix(c_cpu,M,N);
        free(c_cpu);
    }

    free(a);
    free(b);
    free(c);
    free(devices);
    free(chosen);
    free(taskWork);
    free(taskWorkSquared);
    free(success);
    printf("\n");
    return 0;
} // end main
