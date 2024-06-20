#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <assert.h>
#include <math.h>
#include <omp.h>
#include <limits.h>

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

#define MIN(x,y) ((x)<(y)?(x):(y))
#define MAX(x,y) ((x)>(y)?(x):(y))
#define ABS(x) ((x) < 0 ? -(x) : (x))
#define EPSILON 1e-4

using data_type = float;

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

void printMatrix(data_type *mat, int m, int n){
    for(int i=0;i<m;i++){
        for(int j=0;j<n;j++){
            printf("%lf ",mat[i*n+j]);
        }
        printf("\n");
    }
    printf("\n");
}

int main(int argc, char *argv[])
{
    const int ndevs = omp_get_num_devices();
    assert(ndevs > 0);
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
    data_type granularity = 0.5;
    int M=PSIZE,N=PSIZE,K=PSIZE;
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
        if (argc > 4) {
            granularity = atof(argv[4]);
            if (granularity < 0.0 || granularity > 1.0) {
                fprintf(stderr, "Error: granularity must be between 0.0 and 1.0. Received %lf.\n", granularity);
                exit(1); // Exit with error code 1
            }
        }
        // if (argc > 4)
        //     numloop = atoi(argv[4]);
    }
    int a_size = M*K, b_size = K*N, c_size = M*N;

    int rowsPerTask = MAX(1, (1.0 - granularity) * M);
    int numTasks = (M + rowsPerTask - 1) / rowsPerTask;
    printf("bench_works [m=%d] [n=%d] [k=%d] [numTasks=%d] [granularity=%0.2lf] [rowsPerTask=%d] [numThreads=%d] \n", M, N, K, numTasks, granularity, rowsPerTask, numThreads);

    data_type *a = (data_type *)malloc(a_size * sizeof(data_type));
    data_type *b = (data_type *)malloc(b_size * sizeof(data_type));
    data_type *c = (data_type *)malloc(c_size * sizeof(data_type));
    data_type *c_cpu = (data_type *)malloc(c_size * sizeof(data_type));

    int *taskWork = (int *)malloc(sizeof(int) * numTasks);
    int *taskWorkSquared = (int *)malloc(sizeof(int) * numTasks);

    int *chosen = (int *)malloc(sizeof(int) * numTasks);
    int *success = (int *)malloc(sizeof(int) * numTasks);

    // initialize

    for (int i = 0; i < a_size; i++)
        a[i] = (data_type)rand()/RAND_MAX * 2.0 - 1.0;

    for (int i = 0; i < b_size; i++)
        b[i] = (data_type)rand()/RAND_MAX * 2.0 - 1.0;

    for (int i = 0; i < c_size; i++)
        c[i] = 0.0;

    for (int i = 0; i < c_size; i++)
        c_cpu[i] = 0.0;

    int ctaskwork;
//     for (int i = 0; i < numTasks; i++)
//     {
// #ifdef RANDOM_SIZED_TASKS
//         // ctaskwork =  LOWERLT + (rand()%(probSize-LOWERLT) -1);
//         ctaskwork = 1 + (rand() % probSize - 1);
//         // ctaskwork = probSize;
// #else
// #ifdef INCREASING_SIZED_TASKS
//         ctaskwork = 1 + (rand() % probSize - 1);
//         // ctaskwork =  LOWERLT + (rand()%(probSize-LOWERLT) -1);
// #endif
// #endif
// #ifdef INCREASING_SIZED_TASKS
//         int j = i - 1;
//         while ((j >= 0) && (ctaskwork < taskWork[j]))
//         {
//             taskWork[j + 1] = taskWork[j];
//             taskWorkSquared[j + 1] = taskWorkSquared[j];
//             j--;
//         }
//         taskWork[j + 1] = ctaskwork;
//         taskWorkSquared[j + 1] = ctaskwork * ctaskwork;

// #else
//         taskWork[i] = ctaskwork;
//         taskWorkSquared[i] = ctaskwork * ctaskwork;
// #endif
//     }

    // printf("taskWork[] = ");
    // for (int i = 0; i < numTasks; i++)
    //     printf("%d ", taskWork[i]);
    // printf("\n");

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

    int host_id = omp_get_initial_device();
    data_type * __restrict (*dev_pointers)[3];
    dev_pointers = (data_type *(*)[3])malloc(ndevs * sizeof(*dev_pointers));

    
    start_iterations = omp_get_wtime();
    #pragma omp parallel for shared(dev_pointers)
    for(int d=0;d<ndevs;d++){
        // printf("CPU %d\n",omp_get_thread_num());
        dev_pointers[d][0] = (data_type *)omp_target_alloc(a_size * sizeof(data_type), d);
        dev_pointers[d][1] = (data_type *)omp_target_alloc(b_size * sizeof(data_type), d);
        dev_pointers[d][2] = (data_type *)omp_target_alloc(c_size * sizeof(data_type), d);

        omp_target_memcpy(dev_pointers[d][0], a, a_size * sizeof(data_type), 0, 0, d, host_id);
        omp_target_memcpy(dev_pointers[d][1], b, b_size * sizeof(data_type), 0, 0, d, host_id);
        omp_target_memcpy(dev_pointers[d][2], c, c_size * sizeof(data_type), 0, 0, d, host_id);
    }

    #pragma omp parallel
    {
        #pragma omp single
        {
            #pragma omp parallel for shared(success, nextTask, chosen)
            // #pragma omp taskloop shared(success, nextTask, chosen) grainsize(gsz)
            for (int i = 0; i < numTasks; i++)
            {
                // if (taskWork[i] > probSize)
                //     taskWork[i] = probSize;
                // const int NN = taskWork[i];
                // const int NNsq = NN * NN;
                // const int nl = rand() % numloop + 1;
                // printf("i = %d nl = %d\n", i, nl);
                // set up work needed for the firing of task[i],
                // thread picks a device for its current task
                // (or defers the decision by not assigning to chosen[i])

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
/*#pragma omp task depend(in: chosen[i], inout: success[i])// name: fire [i]
      {
    int d = chosen[i]; // assert(0 <= chosen[i] <= ndevs-1)
           if (dev != -1) chosen[i] = dev;
               success[i] = 0;
          // name: fire [i]
*/
                #pragma omp task depend(in : chosen[i]) depend(inout : success[i])
                {
                    int d = chosen[i]; // assert(0 <= chosen[i] <= ndevs-1)
                    data_type * __restrict d_a = dev_pointers[d][0];
                    data_type * __restrict d_b = dev_pointers[d][1];
                    data_type * __restrict d_c = dev_pointers[d][2];
    
                    // #if defined(ASYN)
                    // #pragma omp target device(d) \
                    // map(to : a[0 : a_size], b[0 : b_size]) map(tofrom : success[i : 1], devices[d : 1], taskWork[i : 1], c[0 : c_size]) nowait
                    // #else
                    // #pragma omp target device(d) \
                    // map(to : a[0 : a_size], b[0 : b_size]) map(tofrom : success[i : 1], devices[d : 1], taskWork[i : 1], c[0 : c_size])
                    // #endif
                    #if defined(ASYN)
                    #pragma omp target device(d) \
                    map(tofrom : success[i : 1], devices[d : 1], taskWork[i : 1]) is_device_ptr(d_a,d_b,d_c) nowait
                    #else
                    #pragma omp target device(d) \
                    map(tofrom : success[i : 1], devices[d : 1], taskWork[i : 1]) is_device_ptr(d_a,d_b,d_c)
                    #endif
                    {
                        devices[d]++;
                        // const int NN = taskWork[i];
                        // printf("GPU = %d NN = %d\n", omp_get_device_num(), NN);
                        #ifdef MM
                        printf("GPU %d\n",omp_get_device_num());
                        // for (int l = 0; l < nl; l++)
                            // #pragma omp target teams disrtibute parallel for colapse(3)

                            // printf("GPU %d: task num = %d start = %d end = %d size = %d\n",omp_get_device_num(),i,start,end,nRows);
                            for (int ii = start; ii < end; ii++)
                                for (int jj = 0; jj < N; jj++){
                                    data_type sum = data_type();
                                    for (int kk = 0; kk < K; kk++)
                                        sum += d_a[ii * K + kk] * d_b[kk * N + jj];
                                    d_c[ii * N + jj] = sum;
                                }
                        
                        success[i] = 1; // Note to Mathi: coudl this be outside ifdef?
                        #endif
                    }                    // end target
                }                        // end task
                #pragma omp task depend(in : success[i]) // name: post[i]
                {
                    int d = chosen[i]; // d is the device that just got freed
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
                }
            } // end taskloop
        }     // end of single
        #pragma omp barrier
        
    } // end parallel
    for(int d=0;d<ndevs;d++){
        data_type *d_a = dev_pointers[d][0];
        data_type *d_b = dev_pointers[d][1];
        data_type *d_c = dev_pointers[d][2];
        omp_target_memcpy(c_cpu, d_c, c_size * sizeof(data_type), 0, 0, host_id, d);
        
        #pragma omp parallel for
        for(int i=0;i<c_size;i++){
            // printf("CPU %d\n",omp_get_thread_num());
            c[i] += c_cpu[i];
            c_cpu[i] = 0.0;
        }
        omp_target_free(d_a, d);
        omp_target_free(d_b, d);
        omp_target_free(d_c, d);
    }

    int check = 0;
    end_iterations = omp_get_wtime();
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
    printf("Statistics for %d iterations:\n", numTasks);
    printf("Loop took %lf seconds\n", end_iterations - start_iterations);


    // printf("Total number of CPU threads=%d\n", omp_get_num_threads());
    printf("GPU Done... now checking correctness\n");
    end_iterations = omp_get_wtime();
    for (int ii = 0; ii < M; ii++)
        for (int jj = 0; jj < N; jj++)
            for (int kk = 0; kk < K; kk++)
                c_cpu[ii * N + jj] += a[ii * K + kk] * b[kk * N + jj];
    printf("CPU MM Mul took %lf sec\n",omp_get_wtime()-end_iterations);
    bool flag = true;
    for (int i = 0; i < M; i++)
    {
        for (int j = 0; j < N; j++){
            data_type x = c[i * N + j], y = c_cpu[i * N + j];
            if (x != y && ABS(x - y) > EPSILON) // data_type precision comparision upto 10^-6 for types like doubles
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
    
    // printMatrix(a,M,K);printMatrix(b,K,N);printMatrix(c,M,N);printMatrix(c_cpu,M,N);

    free(a);
    free(b);
    free(c);
    free(c_cpu);
    free(devices);
    free(chosen);
    free(taskWork);
    free(taskWorkSquared);
    free(success);
    return 0;
} // end main
