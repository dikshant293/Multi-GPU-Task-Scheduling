#pragma once
#ifndef UTILS_HPP
#define UTILS_HPP

#define PI                          (3.14159)

#ifndef MAX_PRINT_NEDGE
#define MAX_PRINT_NEDGE             (10000000)
#endif

// Read https://en.wikipedia.org/wiki/Linear_congruential_generator#Period_length
// about choice of LCG parameters
// From numerical recipes
// TODO FIXME investigate larger periods
#define MLCG                        (2147483647)    // 2^31 - 1
#define ALCG                        (16807)         // 7^5
#define BLCG                        (0)

#include <random>
#include <utility>
#include <cstring>
#include <vector>
#include <cmath>
#include <stdint.h>
#include <cuda_runtime.h>

#ifdef USE_32_BIT_GRAPH
using GraphElem = int;
using GraphWeight = float;
#else
using GraphElem = int;
using GraphWeight = float;
#endif

extern unsigned seed;

// Is nprocs a power-of-2?
int is_pwr2(int nprocs)
{ return ((nprocs != 0) && !(nprocs & (nprocs - 1))); }

// return uint32_t seed
GraphElem reseeder(unsigned initseed)
{
    std::seed_seq seq({initseed});
    std::vector<std::uint32_t> seeds(1);
    seq.generate(seeds.begin(), seeds.end());

    return (GraphElem)seeds[0];
}

// CUDA kernel for LCG
__global__ void lcg_kernel(GraphElem *d_thread_seeds, GraphWeight *d_drand, GraphElem n_per_thread, GraphElem n_total, GraphElem num_threads)
{
    GraphElem tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid >= num_threads)
        return;

    // GraphElem ALCG = 16807;      // 7^5
    // GraphElem MLCG = 2147483647; // 2^31 - 1

    GraphElem x = d_thread_seeds[tid];

    GraphElem start_idx = tid * n_per_thread;

    for (GraphElem i = 0; i < n_per_thread; i++)
    {
        if (start_idx + i >= n_total)
            break;

        x = (x * ALCG) % MLCG;

        // Convert to GraphWeight between 0 and 1
        GraphWeight mult = 1.0 / (GraphWeight)(1.0 + (GraphWeight)(MLCG - 1));
        d_drand[start_idx + i] = (GraphWeight)((GraphWeight)abs(x) * mult);
    }
}

// Parallel Linear Congruential Generator
// x[i] = (a*x[i-1] + b)%M
class LCG
{
    public:
        LCG(unsigned seed, GraphWeight* drand,
            GraphElem n, GraphElem num_threads = 1):
        seed_(seed), drand_(drand), n_(n), num_threads_(num_threads)
        {
            x0_ = reseeder(seed_);

            n_per_thread_ = n_ / num_threads_;
            if (n_ % num_threads_ != 0)
                n_per_thread_++; // Ensure all numbers are generated

            thread_seeds_.resize(num_threads_);

            compute_thread_seeds();
        }

        ~LCG() { }

        // Compute per-thread starting seeds
        void compute_thread_seeds()
        {
            GraphElem c = (BLCG * modinv(ALCG - 1, MLCG)) % MLCG;

            for (GraphElem t = 0; t < num_threads_; t++)
            {
                GraphElem exponent = t * n_per_thread_;
                GraphElem a_exp = pow_mod(ALCG, exponent, MLCG);
                GraphElem x_t = (a_exp * x0_ + c * (a_exp - 1)) % MLCG;
                thread_seeds_[t] = x_t;
            }
        }

        GraphElem pow_mod(GraphElem a, GraphElem n, GraphElem M)
        {
            GraphElem result = 1;
            a = a % M;
            while (n > 0)
            {
                if (n % 2 == 1)
                    result = (result * a) % M;
                n = n / 2;
                a = (a * a) % M;
            }
            return result;
        }

        GraphElem modinv(GraphElem a, GraphElem M)
        {
            return pow_mod(a, M - 2, M);
        }

        void generate()
        {
            // Allocate device memory
            GraphElem *d_thread_seeds;
            GraphWeight *d_drand;

            cudaMalloc((void**)&d_thread_seeds, num_threads_ * sizeof(GraphElem));
            cudaMalloc((void**)&d_drand, n_ * sizeof(GraphWeight));

            // Copy thread_seeds_ to device
            cudaMemcpy(d_thread_seeds, thread_seeds_.data(), num_threads_ * sizeof(GraphElem), cudaMemcpyHostToDevice);

            // Launch kernel
            int threads_per_block = 256;
            int num_blocks = (num_threads_ + threads_per_block - 1) / threads_per_block;

            lcg_kernel<<<num_blocks, threads_per_block>>>(d_thread_seeds, d_drand, n_per_thread_, n_, num_threads_);

            // Copy drand_ back to host
            cudaMemcpy(drand_, d_drand, n_ * sizeof(GraphWeight), cudaMemcpyDeviceToHost);

            // Free device memory
            cudaFree(d_thread_seeds);
            cudaFree(d_drand);
        }

        // Rescale the random numbers between lo and hi
        void rescale(GraphWeight* new_drand, GraphElem idx_start, GraphWeight const& lo)
        {
            GraphWeight range = (1.0 / (GraphWeight)num_threads_);

            for (GraphElem i = idx_start, j = 0; i < n_; i++, j++)
                new_drand[j] = lo + (GraphWeight)(range * drand_[i]); // lo-hi
        }

    private:
        unsigned seed_;
        GraphElem n_, x0_;
        GraphElem num_threads_, n_per_thread_;
        GraphWeight* drand_;
        std::vector<GraphElem> thread_seeds_;
};

#endif // UTILS_HPP
