TARGET_SYS=polaris
API=mpi-cuda
SUBCOMPILER=nvhpc

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


#Options for sub-compilers:
# a) llvm-clang
# b) intel-ipc
# c) nvidia-nvhpc
ifeq ($(SUBCOMPILER),clang)
    CFLAGS=-march=native -mp=gpu -std=c++20 -use_fast_math --diag_suppress declared_but_not_referenced -Xlinker -znoexecstack -Xcompiler "-fopenmp,-fpic" -Xnvlink "-suppress-stack-size-warning"
else ifeq ($(SUBCOMPILER),icx)
    CFLAGS=-std=c++20 -O3 -fiopenmp -fopenmp-targets=spir64 -D__STRICT_ANSI__ -xCORE-AVX512  #-qopenmp-simd
else ifeq ($(SUBCOMPILER),nvhpc)
    CFLAGS=-std=c++20 -Wall -O3 -I. -fast -mp=gpu -gpu=cc80 -Xcompiler "-fopenmp,-fpic" -march=native -Xlinker -znoexecstack -O3 -I. -fast  --diag_suppress declared_but_not_referenced --diag_suppress set_but_not_used
endif

CMD=
#Options for programming models: 
# a) mpi-omp
# b) mpi-cuda
ifeq ($(API),mpi-omp)
   SRC=mpi_motionsim.cpp
   OBJ=mpi_motionsim
   CFLAGS+=-DUSEOPENMP -DVECTORIZE
else ifeq ($(API),mpi-cuda)
   SRC=mpi_motionsim.cu
   OBJ=mpi_motionsim
   CMD=cp mpi_motionsim.cpp mpi_motionsim.cu
   CFLAGS+=-DVECTORIZE
endif

OPTIMI=-O3

CUDA_LD = -lcudart -L/usr/local/cuda/lib64
BASE = mpicxx -o ${OBJ} ${SRC} ${OPTIMI} ${CFLAGS}
ifeq ($(API),mpi-cuda)
    BASE+=$(CUDA_LD)
endif

TARGET=mpiexec -n ${TRANKS} -ppn ${RANKS_NODE} --cpu-bind=${CPU_AFFINITY} -d 1 ./${GPU_AFFINITY} ./${OBJ}

all: clean
	$(CMD)
	$(BASE)

n=20000
test: clean all
	$(TARGET) $(n) 0.0 $(n)

clean:
	rm -rf $(OBJ) $(OBJ).cu