#Options for HPC systems:
# a) Polaris
# b) Aurora
ifeq ($(TARGET_SYS),polaris)
  TRANKS=4
  RANKS_NODE=4
  CPU_AFFINITY=list:0-7:8-15:16-23:24-31
  GPU_AFFINITY=affinity_gpu_polaris.sh
else ifeq ($(TARGET_SYS),aurora)
  TRANKS=12
  RANKS_NODE=12
  CPU_AFFINITY=list:0-7:8-15:16-23:24-31:32-39:40-47:52-59:60-67:68-75:76-83:84-91:92-99
  GPU_AFFINITY=affinity_gpu_aurora.sh
endif

#Options for programming models: 
# a) mpi-omp
# b) mpi-cuda
ifeq ($(API),mpi-omp)
   SRC=mpi_cuda_matmul.cpp
   OBJ=mpi_omp_matmul
else ifeq ($(API),mpi-cuda)
   SRC=mpi_cuda_matmul.cu
   OBJ=mpi_cuda_matmul
endif

#Options for sub-compilers:
# a) llvm-clang
# b) intel-ipc
# c) nvidia-nvhpc
ifeq ($(SUBCOMPILER),clang)
    #CFLAGS=-march=native -mp=gpu -std=c++20 -use_fast_math --diag_suppress declared_but_not_referenced -Xlinker -znoexecstack -Xcompiler "-fopenmp,-fpic" -Xnvlink "-suppress-stack-size-warning"
else ifeq ($(SUBCOMPILER),icx)
    CFLAGS=-std=c++20 -fiopenmp -fopenmp-targets=spir64 -D__STRICT_ANSI__ -xCORE-AVX512  #-qopenmp-simd
else ifeq ($(SUBCOMPILER),nvhpc)
    CFLAGS=-march=native -mp=gpu -std=c++20 -use_fast_math --diag_suppress declared_but_not_referenced -Xlinker -znoexecstack -Xcompiler "-fopenmp,-fpic" -Xnvlink "-suppress-stack-size-warning"
endif

OPTIMI=-O3

CUDA_LD = -lcudart -L/usr/local/cuda/lib64
BASE_CPP = mpicxx -o ${OBJ} ${SRC} ${OPTIMI} ${CFLAGS}
BASE_CU = mpicxx -o ${OBJ} ${SRC} ${OPTIMI} ${CFLAGS} ${CUDA_LD}

TARGET=mpiexec -n ${TRANKS} -ppn ${RANKS_NODE} --cpu-bind=${CPU_AFFINITY} -d 1 ./${GPU_AFFINITY} ./${OBJ}

all:
	$(BASE_CPP) -DUSEOPENMP -DVECTORIZE
	@cp mpi_cuda_matmul.cpp mpi_cuda_matmul.cu
	$(BASE_CU) $(CUDA_LD) -DVECTORIZE
	@rm mpi_cuda_matmul.cu

run: all
	$(TARGET) 20 20 20 1

corr:
	@cp mpi_cuda_matmul.cpp mpi_cuda_matmul.cu
	@$(BASE_CU) $(CUDA_LD) 
	$(TARGET) 200 200 200 1
	@$(BASE_CU) $(CUDA_LD) -DVECTORIZE
	$(TARGET) 200 200 200 1
	@rm mpi_cuda_matmul.cu
	@$(BASE_CPP) -DUSEOPENMP
	$(TARGET) 200 200 200 1
	@$(BASE_CPP) -DUSEOPENMP -DVECTORIZE
	$(TARGET) 200 200 200 1

test: 
	@cp mpi_cuda_matmul.cpp mpi_cuda_matmul.cu
	@$(BASE_CU) $(CUDA_LD)
	$(TARGET) 20000 20000 20000
	@$(BASE_CU) $(CUDA_LD) -DVECTORIZE
	$(TARGET) 20000 20000 20000
	@rm mpi_cuda_matmul.cu
	@$(BASE_CPP) -DUSEOPENMP
	$(TARGET) 20000 20000 20000
	@$(BASE_CPP) -DUSEOPENMP -DVECTORIZE
	$(TARGET) 20000 20000 20000

openmp:
	@$(BASE_CPP) -DUSEOPENMP
	$(TARGET) 20000 20000 20000
	@$(BASE_CPP) -DUSEOPENMP -DVECTORIZE
	$(TARGET) 20000 20000 20000