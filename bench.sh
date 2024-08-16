#!/bin/bash

# Get the number of columns in the terminal
width=$(tput cols)

# Generate a line of '~' characters fitting the terminal width
line=$(printf '~%.0s' $(seq 1 $width))

if [[ "$1" == -[pP]* ]]; then
    export TARGET_SYS=polaris
    export COMPILER=llvm_nv
    export SUBCOMPILER=nvhpc
elif [[ "$1" == -[aA]* ]]; then
    export TARGET_SYS=aurora
    export COMPILER=intel
    export SUBCOMPILER=icx
    export API=mpi-omp
    export OMP_TARGET_OFFLOAD=MANDATORY
    export LIBOMPTARGET_PLUGIN=LEVEL0
    #export ZE_AFFINITY_MASK=0
    export EnableImplicitScaling=1
    export LIBOMPTARGET_LEVEL_ZERO_MEMORY_POOL=device,64,128
fi


#ml use /soft/modulefiles
#ml load llvm

#nvc++ --version
#clang++ --version
mpicxx --version

echo passes = $2

for i in $(seq 1 $2) 

# for i in $(seq 1 $2) 
# do
#     echo "$line"
#     echo Pass $i
#     echo 

#     cd cuda/
#     echo CUDA
#     make test -f Makefile.motion

#     echo "$line"
#     echo

#     cd ../omp/
#     echo OPENMP
#     make test -f Makefile.motion

#     cd ../

#     echo "$line"
#     echo "$line"
#     echo
# done

# echo passes = $1

# for i in $(seq 1 $2) 
# do
#     echo "$line"
#     echo Pass $i
#     echo 

#     # cd mpi/
#     # echo MPI
#     # make openmp 
#     # cd ../

#     # echo "$line"
#     # echo

#     cd cuda/
#     echo CUDA
#     make gran-test
#     cd ../

#     echo "$line"
#     echo

#     cd omp/
#     echo OPENMP
#     make gran-test
#     cd ../

#     echo "$line"
#     echo "$line"
#     echo
# done

echo passes = $1

for i in $(seq 1 $2) 
do
    echo "$line"
    echo Pass $i
    echo 

    # cd mpi/
    # echo MPI
    # make openmp 
    # cd ../

    # echo "$line"
    # echo

    cd cuda/
    echo CUDA
    make mem
    cd ../

    echo "$line"
    echo

    # cd omp/
    # echo OPENMP
    # make gran-test
    # cd ../

    # echo "$line"
    echo "$line"
    echo
done