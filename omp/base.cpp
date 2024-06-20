
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <omp.h>
#define MAX_RANK_SIZE 100
#define BATCH_SIZE 10
inline unsigned
gpu_scheduler_rr(unsigned *occupancies, int taskID, int ngpus)
{
   const unsigned chosen = taskID % ngpus;
#pragma omp atomic
   occupancies[chosen]++;
   return chosen;
}
int main(int argc, char **argv)
{

   int s = 7; // rand()%(MAX_RANK_SIZE+1);
   long elems = s * s;
   float *a = (float *)malloc(sizeof(float) * elems);
   float *b = (float *)malloc(sizeof(float) * elems);
   float *c = (float *)malloc(sizeof(float) * elems);

   for (int i = 0; i < elems; i++)
   {
      a[i] = 1.0;
      b[i] = 2.0;
      c[i] = 0.0;
   }
   int sucess[BATCH_SIZE];
   const int ndevs = omp_get_num_devices();
   printf("There are %d GPUs\n", ndevs);

   unsigned *occupancies = NULL;
   occupancies = (unsigned *)calloc(ndevs, sizeof(*occupancies));

   #pragma omp parallel
   {
      printf("cpu %d\n",omp_get_thread_num());
      #pragma omp single
      {
         printf("in single cpu %d\n",omp_get_thread_num());
         #pragma omp taskloop shared(sucess)
         for (int z = 0; z < BATCH_SIZE; z++)
         {
            const int dev = gpu_scheduler_rr(occupancies, z, ndevs);
            // c[z] = 0;
            #pragma omp task depend(out : sucess[z])
            {
               sucess[z] = 0;
            }
            #pragma omp task depend(inout : sucess[z])
            {
               #pragma omp target device(dev) \
               map(to : a[ : elems], b[ : elems]) \
               map(tofrom : sucess[z : 1], c[ : elems], occupancies[dev : 1])
               {
                  printf("gpu %d\n",omp_get_device_num());
                  for (int i = 0; i < s; i++)
                     for (int j = 0; j < s; j++)
                        for (int k = 0; k < s; k++)
                           c[i * s + j] += a[i * s + k] * b[k * s + j];
                  sucess[z] = 1;
               } // end target
            }    // end task
            #pragma omp task depend(in : sucess[z])
            {
               #pragma omp atomic
               occupancies[dev]--;
            }
         } // end taskloop
      }    // end single
   }       // end parallel
   for (int i = 0; i < s; i++){
      for (int j = 0; j < s; j++)
         printf("%0.2f ",c[i * s + j]);
      printf("\n");
   }
   return 1;
}
