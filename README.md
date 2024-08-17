# Multi-GPU-Task-Scheduling

This repository contains implementations of matrix multiplication using multiple GPUs, leveraging different programming models such as CUDA, OpenMP, and MPI. The goal of these implementations is to explore and benchmark the performance of matrix multiplication across various computational strategies and settings.

## Overview

Matrix multiplication, a critical operation in many scientific computations, serves as a standard benchmark for measuring the efficiency and performance of computational environments. In this project, we compare different approaches CUDA, OpenMP and MPI.

The primary focus is on analyzing the performance impact of different scheduling strategies and task granularity levels on the overall computation time and efficiency.

## Installation

To clone and run these implementations, you'll need Git and potentially additional libraries, depending on the programming model used. Follow these steps:

```bash
# Clone this repository
https://github.com/dikshant293/Multi-GPU-Task-Scheduling.git

# Go into the repository
cd Multi-GPU-Task-Scheduling

# For CUDA implementations, ensure you have the CUDA toolkit installed
# For OpenMP implementations, ensure your compiler supports OpenMP
# For MPI implementations, install an MPI library like MPICH or OpenMPI
```

## Compilers
* CUDA - nvc++
* OpenMP - LLVM clang++
* MPI - mpich (openmpi)

## Running

All implementations are accompanied with `Makefile`s for easy running and testing. Running the `make` command compiles the code into an executable. For multiplying a MxK matrix with a KxN matrix:

```bash
./<executableName> <M> <N> <K> <granularity>
```

To reproduce data provided in the paper run:
* CUDA: `make gran-test` and `make cmp`
* OpenMP: `make gran-test`
* MPI: `make test` for MPI+CUDA and `make test API=mpi-omp SUBCOMPILER=clang` for MPI+OMP
