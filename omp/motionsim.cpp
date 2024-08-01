/***************************************************************************
 *
Copyright 2020 Intel Corporation

Permission is hereby granted, free of charge, to any person obtaining a copy of
this software and associated documentation files (the "Software"), to deal in
the Software without restriction, including without limitation the rights to
use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
the Software, and to permit persons to whom the Software is furnished to do so,
subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

 * motionsim.cpp
 *
 * Description:
 *   This code sample will implement a simple example of a Monte Carlo
 *   simulation of the diffusion of water molecules in tissue.
 *
 **************************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <chrono>
#include <cmath>
#include <ctime>
#include <iomanip>
#include <iostream>
#include <vector>
#include <limits.h>
#include <omp.h>
#include <assert.h>

#define CEIL(x, y) (((x) + (y) - 1) / (y))
#define MAX(x, y) (((x) > (y)) ? (x) : (y))
#define MIN(x, y) (((x) < (y)) ? (x) : (y))
#define ABS(x) (((x) < (0)) ? (-x) : (x))

// Helper functions
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

// This function displays correct usage and parameters
void usage(std::string programName)
{
    std::cout << " Incorrect number of parameters " << std::endl;
    std::cout << " Usage: ";
    std::cout << programName << " <Numbeof Iterations> <granularity(default 0.9)>" << std::endl
              << std::endl;
}

// This function prints a 2D matrix
template <typename T>
void print_matrix(T **matrix, size_t size_X, size_t size_Y)
{
    std::cout << std::endl;
    for (size_t i = 0; i < size_X; ++i)
    {
        for (size_t j = 0; j < size_Y; ++j)
        {
            std::cout << std::setw(3) << matrix[i][j] << " ";
        }
        std::cout << std::endl;
    }
}

// This function prints a vector
template <typename T>
void print_vector(T *vector, size_t n)
{
    std::cout << std::endl;
    for (size_t i = 0; i < n; ++i)
    {
        std::cout << vector[i] << " ";
    }
    std::cout << std::endl;
}

auto clk = std::chrono::high_resolution_clock::now();

void start_timer(){
    clk = std::chrono::high_resolution_clock::now();
}

void end_timer(std::string func){
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::high_resolution_clock::now() - clk);
    std::cout<<func<<" took "<<1.0e-9 * duration.count()<<" seconds\n";
}

// Function to calculate chunk sizes from start indices
std::vector<int> calculateChunkSizes(const std::vector<int> &startIndexes, int n)
{
    std::vector<int> chunkSizes;
    for (size_t i = 0; i < startIndexes.size(); ++i)
    {
        if (i == startIndexes.size() - 1)
        {
            chunkSizes.push_back(n - startIndexes[i]); // Last chunk goes to the end of the array
        }
        else
        {
            chunkSizes.push_back(startIndexes[i + 1] - startIndexes[i]);
        }
    }
    return chunkSizes;
}

std::vector<int> generateEqualChunkStartIndices(int n, int m)
{
    std::vector<int> startIndexes;
    int baseSize = n / m;  // Base size of each chunk
    int remainder = n % m; // Remainder to be distributed
    int startIndex = 0;

    // Generate starting indices based on uniform chunk sizes
    for (int i = 0; i < m; ++i)
    {
        startIndexes.push_back(startIndex);
        int currentChunkSize = baseSize + (i < remainder ? 1 : 0); // Distribute remainder among the first few chunks
        startIndex += currentChunkSize;
    }

    return startIndexes;
}

// This function distributes simulation work across workers
void motion_device(float *particleX, float *particleY,
                   float *randomX, float *randomY, int **grid, int grid_size,
                   size_t n_particles, unsigned int nIterations, float radius,
                   size_t *map, float granularity = 0.0)
{

    srand(17);

    // Scale of random numbers
    const size_t scale = 100;

    // Compute vectors of random values for X and Y directions
    for (size_t i = 0; i < n_particles * nIterations; i++)
    {
        randomX[i] = rand() % scale;
        randomY[i] = rand() % scale;
    }

    const size_t MAP_SIZE = n_particles * grid_size * grid_size;


    const int ndevs = omp_get_num_devices();
    assert(ndevs > 0);
    int *devices = (int *)calloc(ndevs, sizeof(*devices));
    double start_iterations, end_iterations;
    unsigned *lastGPU = NULL;

    //  int chosen[N];
    unsigned *occupancies = (unsigned *)calloc(ndevs, sizeof(*occupancies));
    unsigned long *gpuLoad = (unsigned long *)calloc(ndevs, sizeof(*gpuLoad));

    int particlesPerTask = MAX(1, (1.0 - granularity) * n_particles);
    int numTasks = CEIL(n_particles, particlesPerTask);

    int *chosen = (int *)malloc(sizeof(int) * numTasks);
    int *success = (int *)malloc(sizeof(int) * numTasks);
    printf("nTasks = %d particles per task = %d\n",numTasks,particlesPerTask);
    std::vector<int> startIndexes = generateEqualChunkStartIndices(n_particles, numTasks);
    std::vector<int> chunkSizes = calculateChunkSizes(startIndexes, n_particles);

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
    start_timer();
    #pragma omp parallel for shared(success, chosen, startIndexes, chunkSizes) schedule(static,1)
    for (int i = 0; i < numTasks; i++)
    {
        int start = startIndexes[i], end = (i == numTasks - 1 ? n_particles : startIndexes[i + 1]);
        int n_parts = end - start;
        const int NNsq = n_parts;

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
        devices[d]++;

        float *randX = randomX + start * nIterations, *randY = randomY + start * nIterations;
        float *partX = particleX + start, *partY = particleY + start;
        size_t *mp_ptr =  map + start * grid_size * grid_size;

        #pragma omp target teams distribute parallel for simd num_teams(CEIL(n_parts,1024)) thread_limit(1024) schedule (static, 1) device(d) \
        map(to : randX[0 : n_parts * nIterations], randY[0 : n_parts * nIterations]) \
        map(tofrom : partX[0 : n_parts], partY[0 : n_parts], mp_ptr[0 : n_parts * grid_size * grid_size])
        for (int ii = start; ii < end; ii++)
        {

            size_t randnumX = 0;
            size_t randnumY = 0;
            float displacementX = 0.0f;
            float displacementY = 0.0f;

            // Start iterations
            // Each iteration:
            //  1. Updates the position of all water molecules
            //  2. Checks if water molecule is inside a cell or not.
            //  3. Updates counter in cells array
            size_t iter = 0;
            while (iter < nIterations)
            {
                // Computes random displacement for each molecule
                // This example shows random distances between
                // -0.05 units and 0.05 units in both X and Y directions
                // Moves each water molecule by a random vector
                const int idx = ii - start;
                randnumX = randX[idx * nIterations + iter];
                randnumY = randY[idx * nIterations + iter];

                // Transform the scaled random numbers into small displacements
                displacementX = (float)randnumX / 1000.0f - 0.0495f;
                displacementY = (float)randnumY / 1000.0f - 0.0495f;

                // Move particles using random displacements
                partX[idx] += displacementX;
                partY[idx] += displacementY;

                // Compute distances from particle position to grid point
                float dX = partX[idx] - truncf(partX[idx]);
                float dY = partY[idx] - truncf(partY[idx]);

                // Compute grid point indices
                int iX = floorf(partX[idx]);
                int iY = floorf(partY[idx]);

                // Check if particle is still in computation grid
                if ((partX[idx] < grid_size) &&
                    (partY[idx] < grid_size) && (partX[idx] >= 0) &&
                    (partY[idx] >= 0))
                {
                    // Check if particle is (or remained) inside cell.
                    // Increment cell counter in map array if so
                    if ((dX * dX + dY * dY <= radius * radius))
                    {
                        // The map array is organized as (particle, y, x)
                        mp_ptr[idx * grid_size * grid_size + iY * grid_size + iX]++;
                    }
                }

                iter++;

            } // Next iteration
        }
    }


// #pragma omp target map(to : randomX[0 : n_particles * nIterations], randomY[0 : n_particles * nIterations]) \
//     map(tofrom : particleX[0 : n_particles], particleY[0 : n_particles], map[0 : MAP_SIZE])
//     {
// #pragma omp teams distribute parallel for simd thread_limit(256)
//         for (int ii = 0; ii < n_particles; ii++)
//         {

//             size_t randnumX = 0;
//             size_t randnumY = 0;
//             float displacementX = 0.0f;
//             float displacementY = 0.0f;

//             // Start iterations
//             // Each iteration:
//             //  1. Updates the position of all water molecules
//             //  2. Checks if water molecule is inside a cell or not.
//             //  3. Updates counter in cells array
//             size_t iter = 0;
//             while (iter < nIterations)
//             {
//                 // Computes random displacement for each molecule
//                 // This example shows random distances between
//                 // -0.05 units and 0.05 units in both X and Y directions
//                 // Moves each water molecule by a random vector

//                 randnumX = randomX[ii * nIterations + iter];
//                 randnumY = randomY[ii * nIterations + iter];

//                 // Transform the scaled random numbers into small displacements
//                 displacementX = (float)randnumX / 1000.0f - 0.0495f;
//                 displacementY = (float)randnumY / 1000.0f - 0.0495f;

//                 // Move particles using random displacements
//                 particleX[ii] += displacementX;
//                 particleY[ii] += displacementY;

//                 // Compute distances from particle position to grid point
//                 float dX = particleX[ii] - truncf(particleX[ii]);
//                 float dY = particleY[ii] - truncf(particleY[ii]);

//                 // Compute grid point indices
//                 int iX = floorf(particleX[ii]);
//                 int iY = floorf(particleY[ii]);

//                 // Check if particle is still in computation grid
//                 if ((particleX[ii] < grid_size) &&
//                     (particleY[ii] < grid_size) && (particleX[ii] >= 0) &&
//                     (particleY[ii] >= 0))
//                 {
//                     // Check if particle is (or remained) inside cell.
//                     // Increment cell counter in map array if so
//                     if ((dX * dX + dY * dY <= radius * radius))
//                     {
//                         // The map array is organized as (particle, y, x)
//                         map[ii * grid_size * grid_size + iY * grid_size + iX]++;
//                     }
//                 }

//                 iter++;

//             } // Next iteration
//         }
//     }

    end_timer("motion sim");
    // For every cell in the grid, add all the counters from different
    // particles (workers) which are stored in the 3rd dimension of the 'map'
    // array
    for (size_t i = 0; i < n_particles; ++i)
    {
        for (size_t y = 0; y < grid_size; y++)
        {
            for (size_t x = 0; x < grid_size; x++)
            {
                if (map[i * grid_size * grid_size + y * grid_size + x] > 0)
                {
                    grid[y][x] += map[i * grid_size * grid_size + y * grid_size + x];
                }
            }
        }
    } // End loop for number of particles

} // End of function motion_device()

int main(int argc, char *argv[])
{
    // Cell and Particle parameters
    const size_t grid_size = 21;       // Size of square grid
    const size_t n_particles = 147456; // Number of particles
    const float radius = 0.5;          // Cell radius = 0.5*(grid spacing)

    // Default number of operations
    size_t nIterations = 50;
    float granularity = 0.9;
    // Read command-line arguments
    try
    {
        nIterations = std::stoi(argv[1]);
    }

    catch (...)
    {
        usage(argv[0]);
        return 1;
    }

    if(argc>2){
        granularity = atof(argv[2]);
        if(granularity<0 or granularity>1){
            usage(argv[0]);
            return 1;
        }
    }

    // Allocate arrays

    // Stores a grid of cells
    int **grid = new int *[grid_size];
    for (size_t i = 0; i < grid_size; i++)
        grid[i] = new int[grid_size];

    // Stores all random numbers to be used in the simulation
    float *randomX = new float[n_particles * nIterations];
    float *randomY = new float[n_particles * nIterations];

    // Stores X and Y position of particles in the cell grid
    float *particleX = new float[n_particles];
    float *particleY = new float[n_particles];

    // 'map' array replicates grid to be used by each particle
    const size_t MAP_SIZE = n_particles * grid_size * grid_size;
    size_t *map = new size_t[MAP_SIZE];

    // Initialize arrays
    for (size_t i = 0; i < n_particles; i++)
    {
        // Initial position of particles in cell grid
        particleX[i] = 10.0;
        particleY[i] = 10.0;

        for (size_t y = 0; y < grid_size; y++)
        {
            for (size_t x = 0; x < grid_size; x++)
            {
                map[i * grid_size * grid_size + y * grid_size + x] = 0;
            }
        }
    }

    for (size_t y = 0; y < grid_size; y++)
    {
        for (size_t x = 0; x < grid_size; x++)
        {
            grid[y][x] = 0;
        }
    }

    // Start timers
    auto start = std::chrono::steady_clock::now();

    // Call simulation function
    motion_device(particleX, particleY, randomX, randomY, grid, grid_size,
                  n_particles, nIterations, radius, map, granularity);

    auto end = std::chrono::steady_clock::now();
    auto time = std::chrono::duration_cast<std::chrono::milliseconds>(end - start)
                    .count();
    std::cout << std::endl;
    std::cout << "Time: " << time << " ms " << std::endl;
    std::cout << std::endl;

    // Displays final grid only if grid small.
    if (grid_size <= 64)
    {
        std::cout << "\n ********************** OUTPUT GRID: " << std::endl;
        print_matrix<int>(grid, grid_size, grid_size);
    }

    // Cleanup
    for (size_t i = 0; i < grid_size; i++)
        delete grid[i];

    delete[] grid;
    delete[] particleX;
    delete[] particleY;
    delete[] randomX;
    delete[] randomY;
    delete[] map;

    return 0;
}