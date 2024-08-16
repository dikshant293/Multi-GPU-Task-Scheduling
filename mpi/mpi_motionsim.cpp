#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
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
#include <mpi.h>

#if defined(USEOPENMP)
#include <omp.h>
#else
#include <cuda.h>
#include <cuda_runtime.h>
#endif

#if defined(VECTORIZE)
#include <vector_types.h>
#endif

#define CEIL(x, y) (((x) + (y) - 1) / (y))
#define MAX(x, y) (((x) > (y)) ? (x) : (y))
#define MIN(x, y) (((x) < (y)) ? (x) : (y))
#define ABS(x) (((x) < (0)) ? (-x) : (x))
#define MAX_TPB 32

#define EPSILON 1e-4

#define PSIZE 2000

#if not defined(USEOPENMP)

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

__global__ void kernel(float *randX, float *randY, float *partX, float *partY, int *mp_ptr, int n_parts, int nIterations, int grid_size, float radius)
{
    for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < n_parts; idx += blockDim.x * gridDim.x)
    {
        // printf("idx = %d nparts = %d\n", idx, n_parts);
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

#endif

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
    for (int i = 0; i < startIndexes.size(); ++i) {
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

// This function displays correct usage and parameters
void usage(std::string programName)
{
    std::cout << " Incorrect number of parameters " << std::endl;
    std::cout << " Usage: ";
    std::cout << programName << " <Numbeof Iterations> <granularity(default 0.9)>" << std::endl
              << std::endl;
}

template <typename T>
void print_checksum(T **matrix, size_t size_X, size_t size_Y)
{
    long long x = 0;
    for (size_t i = 0; i < size_X; ++i)
    {
        for (size_t j = 0; j < size_Y; ++j)
        {
            x += matrix[i][j];
        }
    }
    std::cout <<"Checksum = " << x << std::endl;
}

int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);

    int world_size, world_rank;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    // printf("world size %d rank %d GPUs %d\n",world_size,world_rank,ndevs);

   // Cell and Particle parameters
    const int grid_size = 21;    // Size of square grid
    int n_particles = 2e4; // Number of particles
    const float radius = 0.5;       // Cell radius = 0.5*(grid spacing)

    // Default number of operations
    int nIterations = 50;
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

    if (argc > 2)
    {
        granularity = atof(argv[2]);
        if (granularity < 0 or granularity > 1)
        {
            usage(argv[0]);
            return 1;
        }
    }
    if (argc > 3)
    {
        n_particles = atoi(argv[3]);
    }

    int numRowsPerRank = CEIL(n_particles,world_size);
    
    std::vector<int> startIndexes = generateEqualChunkStartIndices(n_particles, world_size);;
    std::vector<int> chunkSizes = calculateChunkSizes(startIndexes, n_particles);

    checkMPIError(MPI_Barrier(MPI_COMM_WORLD));
        
    int start = startIndexes[world_rank], end = (world_rank==world_size-1 ? n_particles : startIndexes[world_rank+1]);
    int n_parts = end-start;
    
    MPI_Status stat;

    int **grid;
    float *randomX, *randomY, *particleX, *particleY;
    int MAP_SIZE = n_particles * grid_size * grid_size;
    int *map;
    if (world_rank == 0) {
        // Stores a grid of cells
        grid = (int**)malloc(grid_size * sizeof(int*));
        grid[0] = (int*)malloc(grid_size*grid_size * sizeof(int));
        for (int i = 0; i < grid_size; i++)
            grid[i] = grid[0] + grid_size*i;

        #if defined(USEOPENMP)

        // Stores all random numbers to be used in the simulation
        randomX = (float*)malloc(n_particles*nIterations*sizeof(float));
        randomY = (float*)malloc(n_particles*nIterations*sizeof(float));

        // Stores X and Y position of particles in the cell grid
        particleX = (float*)malloc(n_particles*sizeof(float));
        particleY = (float*)malloc(n_particles*sizeof(float));

        // 'map' array replicates grid to be used by each particle
        map = (int*)malloc(MAP_SIZE*sizeof(int));
        
        #else
        // Stores all random numbers to be used in the simulation
        checkCuda(cudaMallocHost(&randomX,n_particles*nIterations*sizeof(float)));
        checkCuda(cudaMallocHost(&randomY,n_particles*nIterations*sizeof(float)));

        // Stores X and Y position of particles in the cell grid
        checkCuda(cudaMallocHost(&particleX,n_particles*sizeof(float)));
        checkCuda(cudaMallocHost(&particleY,n_particles*sizeof(float)));

        // 'map' array replicates grid to be used by each particle
        checkCuda(cudaMallocHost(&map,MAP_SIZE*sizeof(int)));
        #endif

        printf("nparticles = %d niterations = %d granularity = %lf\n",n_particles,nIterations,granularity);
        std::cout<<"total memory = "<<(float)(2 * n_particles * (nIterations + 1) * sizeof(float) + n_particles * grid_size * grid_size * sizeof(int))/1024/1024/1024<<" GB"<<std::endl;
        
        // Initialize arrays
        for (int i = 0; i < n_particles; i++)
        {
            // Initial position of particles in cell grid
            particleX[i] = 10.0;
            particleY[i] = 10.0;

            for (int y = 0; y < grid_size; y++)
            {
                for (int x = 0; x < grid_size; x++)
                {
                    map[i * grid_size * grid_size + y * grid_size + x] = 0;
                }
            }
        }

        for (int y = 0; y < grid_size; y++)
        {
            for (int x = 0; x < grid_size; x++)
            {
                grid[y][x] = 0;
            }
        }

        srand(17);

        // Scale of random numbers
        const int scale = 100;

        // Compute vectors of random values for X and Y directions
        for (int i = 0; i < n_particles * nIterations; i++)
        {
            randomX[i] = rand() % scale;
            randomY[i] = rand() % scale;
        }

        #if defined(VECTORIZE)
        printf("vectorized\n");
        #else
        printf("non vectorized\n");   
        #endif
        #if defined(USEOPENMP)
        printf("using OPENMP target offload\n");
        #else
        printf("using CUDA\n");
        #endif

        start_timer();
    }

    float *randX,*randY,*partX,*partY;
    int *mp_ptr;
    if(world_rank==0){
        randX = randomX + start * nIterations;
        randY = randomY + start * nIterations;
        partX = particleX + start;
        partY = particleY + start;
        mp_ptr =  map + start * grid_size * grid_size;

        for(int i=1;i<world_size;i++){
            int send_start = startIndexes[i], send_end = (i == world_size - 1 ? n_particles : startIndexes[i + 1]);
            int send_n_parts = send_end - send_start;
            checkMPIError(MPI_Send(randomX + send_start * nIterations, send_n_parts * nIterations, MPI_FLOAT, i, 1, MPI_COMM_WORLD));
            checkMPIError(MPI_Send(randomY + send_start * nIterations, send_n_parts * nIterations, MPI_FLOAT, i, 2, MPI_COMM_WORLD));
            checkMPIError(MPI_Send(particleX + send_start, send_n_parts, MPI_FLOAT, i, 3, MPI_COMM_WORLD));
            checkMPIError(MPI_Send(particleY + send_start, send_n_parts, MPI_FLOAT, i, 4, MPI_COMM_WORLD));
            checkMPIError(MPI_Send(map + send_start * grid_size * grid_size, send_n_parts * grid_size * grid_size, MPI_INT, i, 5, MPI_COMM_WORLD));
        }
        
    }
    else{
        #if defined(USEOPENMP)

        // Stores all random numbers to be used in the simulation
        randX = (float*)malloc(n_parts*nIterations*sizeof(float));
        randY = (float*)malloc(n_parts*nIterations*sizeof(float));

        // Stores X and Y position of particles in the cell grid
        partX = (float*)malloc(n_parts*sizeof(float));
        partY = (float*)malloc(n_parts*sizeof(float));

        // 'map' array replicates grid to be used by each particle
        mp_ptr = (int*)malloc(n_parts * grid_size * grid_size*sizeof(int));
        
        #else

        // Stores all random numbers to be used in the simulation
        checkCuda(cudaMallocHost(&randX,n_parts*nIterations*sizeof(float)));
        checkCuda(cudaMallocHost(&randY,n_parts*nIterations*sizeof(float)));

        // Stores X and Y position of particles in the cell grid
        checkCuda(cudaMallocHost(&partX,n_parts*sizeof(float)));
        checkCuda(cudaMallocHost(&partY,n_parts*sizeof(float)));

        // 'map' array replicates grid to be used by each particle
        checkCuda(cudaMallocHost(&mp_ptr,n_parts * grid_size * grid_size*sizeof(int)));
        #endif

        checkMPIError(MPI_Recv(randX,n_parts*nIterations,MPI_FLOAT,0,1,MPI_COMM_WORLD,&stat));
        checkMPIError(MPI_Recv(randY,n_parts*nIterations,MPI_FLOAT,0,2,MPI_COMM_WORLD,&stat));
        checkMPIError(MPI_Recv(partX,n_parts,MPI_FLOAT,0,3,MPI_COMM_WORLD,&stat));
        checkMPIError(MPI_Recv(partY,n_parts,MPI_FLOAT,0,4,MPI_COMM_WORLD,&stat));
        checkMPIError(MPI_Recv(mp_ptr,n_parts * grid_size * grid_size,MPI_INT,0,5,MPI_COMM_WORLD,&stat));   
    }
    // std::cout<<world_rank<<" check 1"<<std::endl;
    #if defined(USEOPENMP)
    #pragma omp target teams distribute parallel for num_teams(CEIL(n_parts,1024)) thread_limit(1024) schedule (static, 1) \
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
    #else
    float *d_randX,*d_randY,*d_partX,*d_partY;
    int *d_mp_ptr;

    cudaMalloc(&d_randX, n_parts * nIterations * sizeof(float));
    cudaMemcpy(d_randX, randX, n_parts * nIterations * sizeof(float), cudaMemcpyHostToDevice);

    cudaMalloc(&d_randY, n_parts * nIterations * sizeof(float));
    cudaMemcpy(d_randY, randY, n_parts * nIterations * sizeof(float), cudaMemcpyHostToDevice);

    cudaMalloc(&d_partX, n_parts * sizeof(float));
    cudaMemcpy(d_partX, partX, n_parts * sizeof(float), cudaMemcpyHostToDevice);

    cudaMalloc(&d_partY, n_parts * sizeof(float));
    cudaMemcpy(d_partY, partY, n_parts * sizeof(float), cudaMemcpyHostToDevice);

    cudaMalloc(&d_mp_ptr, n_parts * grid_size * grid_size * sizeof(int));
    cudaMemcpy(d_mp_ptr, mp_ptr, n_parts * grid_size * grid_size * sizeof(int), cudaMemcpyHostToDevice);

    int tpb = MIN(MAX_TPB, n_parts);
    int blks = CEIL(n_parts, tpb);
    kernel<<<blks, tpb>>>(d_randX, d_randY, d_partX, d_partY, d_mp_ptr, n_parts, nIterations, grid_size, radius);

    cudaMemcpy(partX, d_partX, n_parts * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(partY, d_partY, n_parts * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(mp_ptr, d_mp_ptr, n_parts * grid_size * grid_size * sizeof(int), cudaMemcpyDeviceToHost);

    cudaFree(d_randX);
    cudaFree(d_randY);
    cudaFree(d_partX);
    cudaFree(d_partY);
    cudaFree(d_mp_ptr);

    cudaDeviceSynchronize();
    #endif
    
    // std::cout<<world_rank<<" check 2"<<std::endl;

    if(world_rank==0){
        for(int i=1;i<world_size;i++){
            int recv_start = startIndexes[i], recv_end = (i==world_size-1 ? n_particles : startIndexes[i+1]);
            int recv_n_parts = recv_end-recv_start;
            checkMPIError(MPI_Recv(particleX+recv_start, recv_n_parts,MPI_FLOAT,i,6,MPI_COMM_WORLD,&stat));
            checkMPIError(MPI_Recv(particleY+recv_start, recv_n_parts,MPI_FLOAT,i,7,MPI_COMM_WORLD,&stat));
            checkMPIError(MPI_Recv(map + recv_start * grid_size * grid_size, recv_n_parts * grid_size * grid_size,MPI_INT,i,8,MPI_COMM_WORLD,&stat));
        }
    }
    else{
        checkMPIError(MPI_Send(partX,n_parts,MPI_FLOAT,0,6,MPI_COMM_WORLD));
        checkMPIError(MPI_Send(partY,n_parts,MPI_FLOAT,0,7,MPI_COMM_WORLD));
        checkMPIError(MPI_Send(mp_ptr,n_parts * grid_size * grid_size,MPI_INT,0,8,MPI_COMM_WORLD));
    }
    
    // std::cout<<world_rank<<" check 3"<<std::endl;
    if(world_rank==0){
        end_timer("motion sim");
        for (int i = 0; i < n_particles; ++i){
            for (int y = 0; y < grid_size; y++){
                for (int x = 0; x < grid_size; x++){
                    if (map[i * grid_size * grid_size + y * grid_size + x] > 0){
                        grid[y][x] += map[i * grid_size * grid_size + y * grid_size + x];
                    }
                }
            }
        } // End loop for number of particles

        print_checksum<int>(grid, grid_size, grid_size);
        
        #if defined(USEOPENMP)
        free(grid[0]);
        free(grid);
        free(particleX);
        free(particleY);
        free(randomX);
        free(randomY);
        free(map);
        #else
        cudaFreeHost(grid[0]);
        cudaFreeHost(grid);
        cudaFreeHost(particleX);
        cudaFreeHost(particleY);
        cudaFreeHost(randomX);
        cudaFreeHost(randomY);
        cudaFreeHost(map);
        #endif
    }
    else{
        #if defined(USEOPENMP)
        free(partX);
        free(partY);
        free(randX);
        free(randY);
        free(mp_ptr);
        #else
        cudaFreeHost(partX);
        cudaFreeHost(partY);
        cudaFreeHost(randX);
        cudaFreeHost(randY);
        cudaFreeHost(mp_ptr);
        #endif
    }

    MPI_Finalize();
    return 0;
}
