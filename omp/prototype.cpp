#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <omp.h>
#include <limits.h>
#include <chrono>
#include <sys/time.h>
#include <string.h>

#ifdef MPI
#include <mpi.h>
#endif

/*
#define MAX_RANK_SIZE 100
#define MAX_LOOP 10
#define BATCH_SIZE 100
*/
#define THRESHOLD 100000

/*-----------------------
 * Scheduling Techniques *
 -----------------------*/

/*
 * Schedule a task to a GPU
 * - a GPU is chosen in round robin manner
 */
inline unsigned
gpu_scheduler_rr(unsigned *occupancies, int taskID, int ngpus)
{
  const unsigned chosen = taskID % ngpus;
  #pragma omp atomic
  occupancies[chosen]++;
  return chosen;
}

/*
 * Schedule a task to a GPU based on dynamic load balancing (adaptive) scheduling
 * - a GPU is chosen based on the work load assigned to GPUs  
 */
inline unsigned 
gpu_scheduler_dynamic_lb(unsigned long *gpuLoad, int offset, int ngpus, int taskWeight)
{
  short looking = 1;
  unsigned chosen;
  while (looking) {
    unsigned occ_i;
    unsigned long load;
    unsigned long min_load = ULLONG_MAX;
    for (unsigned i = offset; i < offset+ngpus; i++) {
      #pragma omp atomic read
      load = gpuLoad[i];
      if ( load < min_load ){
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

/*
 * Multi-queue scheduling; a group of tasks can be mapped to a set of GPUs.
 * - static selection of resource set (?)
 * - based on the dynamic load balancing (adaptive) scheduling
 * */
inline unsigned
gpu_scheduler_dynamic_mqlb(unsigned long **gpuLoad, int ngpus, int taskID, int taskWeight, int numQueues)
{  
  int queueID = (taskID*numQueues)/numTasks;
  int ngpusPerQueue = ngpus/numQueues;
  return gpu_scheduler_dynamic_lb(gpuLoad, queueID*ngpusPerQueue, ngpusPerQueue, taskWeight);
}


/*
 * Schedule a task to multi GPUs by decomposing to sub-tasks; a task shares multi GUPs.
 * - based on the round robin scheduling
 * */
inline unsigned
gpu_scheduler_shared(unsigned *occupancies, int taskID, int taskWeight, int ngpus, unsigned *devList, int *taskWorkList)
{
  int weight = taskWeight;
  int subWeight;
  int ndev=0;
  int offset = taskID*ngpus;
  do{
        subWeight = (weight <THRESHOLD) ? weight : THRESHOLD;
        #pragma omp atomic
        taskWorkList[offset+ndev] += subWeight;
        devList[ndev++] = gpu_scheduler_rr(occupancies, taskID, ngpus);
        weight -= THRESHOLD;
    }while( (weight>0) && (ngpus>ndev) );

 return ndev;
}

/*
 * Schedule a task to multi GPUs by decomposing to sub-tasks; a task shares multi GUPs.
 * - based on the dynamic load balancing (adaptive) scheduling
 * */
inline unsigned
gpu_scheduler_shared(unsigned long *gpuLoad, int taskID, int taskWeight, int ngpus, unsigned *devList, int *taskWorkList)
{
  int weight = taskWeight;
  int subWeight;
  int ndev=0;
  int offset = taskID*ngpus;
  do{
        subWeight = (weight <THRESHOLD) ? weight : THRESHOLD;
        #pragma omp atomic
        taskWorkList[offset+ndev] += subWeight;
        devList[ndev++] = gpu_scheduler_dynamic_lb(gpuLoad, 0, ngpus, subWeight);
        weight -= THRESHOLD;
    }while( (weight>0) && (ngpus>ndev) );

 return ndev;
}

/*------------------------
 * Methods to obtain times
 *------------------------*/

template <typename clock, typename startTime, typename endTime>
double elapsed_seconds(std::chrono::time_point<clock, startTime> start,
                       std::chrono::time_point<clock, endTime> end)
{
  using FloatingPointSeconds = std::chrono::duration<double, std::ratio<1>>;
  return std::chrono::duration_cast<FloatingPointSeconds>(end - start).count();
}

template<typename T>
inline double seconds_since(T& time_start)
{
   struct timeval time_end;
   gettimeofday(&time_end,NULL);
   double num_sec     = time_end.tv_sec  - time_start.tv_sec;
   double num_usec    = time_end.tv_usec - time_start.tv_usec;
   return (num_sec + (num_usec/1000000));
}

template<typename T>
inline void start_timer(T& time_start)
{
   gettimeofday(&time_start,NULL);
}

/*
template <typename clock, typename startTime>
inline auto time_since(std::chrono::time_point<clock, startTime> start)
{
   auto now = std::chrono::high_resolution_clock::now();
   using FloatingPointSeconds = std::chrono::duration<double, std::ratio<1>>;
   return std::chrono::duration_cast<FloatingPointSeconds>(now - start).count();
}
*/

int main(int argc, char** argv){

   int gs = 1;
   int MAX_RANK_SIZE = 100;
   int MAX_LOOP = 10;
   int BATCH_SIZE = 100;

   if(argc > 1) gs = atoi(argv[1]);
   if(argc > 2) BATCH_SIZE = atoi(argv[2]);
   if(argc > 3) MAX_RANK_SIZE = atoi(argv[3]);
   if(argc > 4) MAX_LOOP = atoi(argv[4]);
/*
   int tempint;
   for (int i=1; i<argc-1; i+=2)
   {
      if (strcmp("-gs", argv[i]) == 0)
      {
         sscanf(argv[i+1], "%d", &tempint);
         if (tempint >= 1)
            gs  = (int) tempint;
         else
	    printf("Warning: value of -gs argument ignored. Value must be an integer grater than 0 \n ");
      }
   }
*/

   double *A[BATCH_SIZE], *B[BATCH_SIZE], *C[BATCH_SIZE];
   int size[BATCH_SIZE];

   #ifdef MPI
   int process_Rank, size_Of_Cluster;
   MPI_Init(&argc, &argv);
   MPI_Comm_size(MPI_COMM_WORLD, &size_Of_Cluster);
   MPI_Comm_rank(MPI_COMM_WORLD, &process_Rank);
   #endif

   unsigned long *numit = (unsigned long *) malloc(BATCH_SIZE*sizeof(unsigned long));

   for(int z=0; z<BATCH_SIZE; z++) {
       
      size[z] = rand()%MAX_RANK_SIZE+3;
      unsigned long elems = size[z]*size[z];
      A[z] = (double *) malloc(elems*sizeof(double));
      B[z] = (double *) malloc(elems*sizeof(double));
      C[z] = (double *) malloc(elems*sizeof(double));
   }

   for(int z=0; z<BATCH_SIZE; z++) {
      unsigned long elems = size[z]*size[z];
//      printf("batch: %d \t numelems: %lu  \n", z, elems); 
      for(int i = 0; i< elems; i++) 
      {
         A[z][i] = 1.0;
         B[z][i] = 1.0;
         C[z][i] = 0.0;
      }
   }


   int sucess[BATCH_SIZE];
   #ifdef MPI
   const int ndevs = 1;
   #else
   const int ndevs = omp_get_num_devices();
   #endif

   printf("There are %d GPUs\n", ndevs);
   
   int *devices = NULL;
   devices = (int *) calloc(BATCH_SIZE, sizeof(*devices));
 
   unsigned *occupancies = NULL;
   occupancies = (unsigned *) calloc(ndevs, sizeof(*occupancies));

   unsigned long *gpuLoad = NULL;
   gpuLoad  = (unsigned long*) calloc(ndevs, sizeof(*gpuLoad));
 
   //auto time_st = std::chrono::high_resolution_clock::now();
   struct timeval exec_timer;
   start_timer(exec_timer);
   //double cpu_time = -omp_get_wtime();
#pragma omp parallel
{
   #pragma omp single
   {
      #pragma omp taskloop grainsize(gs) shared(sucess)
      for (int z=0; z<BATCH_SIZE; z++)
      {
//         const int dev = gpu_scheduler_rr(occupancies, z, ndevs);
         unsigned long elems = size[z]*size[z];
	 const int dev = gpu_scheduler_dynamic_lb(gpuLoad, 0, ndevs, elems);
    
         devices[z] = dev;
         #pragma omp task depend(out: sucess[z])
         {
            sucess[z] = 0;
         }
         #pragma omp task depend(inout: sucess[z])
         {

	    unsigned long s = size[z];
	    unsigned long elems = size[z]*size[z];
     	    double *a = A[z], *b = B[z], *c = C[z];
	    unsigned long n = rand()%MAX_LOOP+1;
	    numit[z] = n;
	    //#pragma omp atomic
	    //	 gpuLoad[dev] += (n-1)*elems;
            #pragma omp target device(dev)\
	    map(to: a[:elems], b[:elems])\
	    map(tofrom: sucess[z:1], c[:elems], gpuLoad[dev:1]) nowait 
	    //map(tofrom: sucess[z:1], c[:elems], occupancies[dev:1])
            {
	       for(int l = 0; l < n; l++)
	       for (int i = 0; i < s; i++)
        	   for (int j = 0; j < s; j++)
           	   for (int k = 0; k < s; k++)
               		c[i * s + j] += a[i * s + k] * b[k * s + j];
	       sucess[z] = 1;
	    } // end target	
         } // end task
         #pragma omp task depend(in: sucess[z])
         {
            #pragma omp atomic
	    gpuLoad[dev] -= elems;
	    //occupancies[dev]--;
         }
      } //end taskloop
   } //end single
} //end parallel
   //cpu_time += omp_get_wtime();
   double runtime = seconds_since(exec_timer);
   //auto time_end = std::chrono::high_resolution_clock::now();
   //auto elapsed_time = std::chrono::duration_cast<std::chrono::nanoseconds>(time_end -time_st);
   //auto elapsed_time = time_since<std::chrono::time_point>(time_end -time_st);
   //auto elapsed_time = time_since<std::chrono::seconds, std::chrono::time_point>(time_st);
   //printf(" Elapsed time: %.3f seconds. \n", elapsed_time.count()*1e-9 );
   //printf(" CPU time: %.3f seconds. \n", cpu_time );
   printf(" Run time: %.3f seconds. \n", runtime );

   bool fail = false;
   //printf("Batch ID \t Batch Size \t #Iterations \t Device ID \n"); 
   for(int z=0;z< BATCH_SIZE; z++) {
      for (int i = 0; i < size[z]; i++){
        for (int j = 0; j < size[z]; j++) {
	  // printf(" c[%d, %d] = %f \t",i, j, C[z][i * size[z] + j]);
           if((unsigned long) C[z][i * size[z] + j] != (numit[z]*size[z])) {
              fail = true;
              break;
             }
         }
        // printf("\n");
       }
     // printf("%d \t %d \t %lu \t GPU_%d \n", z, (size[z]*size[z]), numit[z], devices[z]); 
      if(fail) break;
    } 

   for(int z=0; z<BATCH_SIZE; z++) {
      free(A[z]);
      free(B[z]);
      free(C[z]);
   }

   free(numit);
   free(occupancies);
   free(gpuLoad);
   free(devices);
   
   #ifdef MPI
   MPI_Finalize();
   #endif

   if(!fail)
     printf("\nSuccess!");
   else
     printf("\nFail");
   
   return 1;
}
