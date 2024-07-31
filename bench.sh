#!/bin/bash

# Get the number of columns in the terminal
width=$(tput cols)

# Generate a line of '~' characters fitting the terminal width
line=$(printf '~%.0s' $(seq 1 $width))

ml use /soft/modulefiles
ml load llvm

nvc++ --version
clang++ --version
mpicxx --version

echo passes = $1

for i in $(seq 1 $1)
do
    echo "$line"
    echo Pass $i
    echo 

    cd mpi/
    echo MPI
    make test 

    echo "$line"
    echo

    cd ../cuda/
    echo CUDA
    make cmp-sch

    echo "$line"
    echo

    cd ../omp/
    echo OPENMP
    make cmp-sch

    cd ../

    echo "$line"
    echo "$line"
    echo
done