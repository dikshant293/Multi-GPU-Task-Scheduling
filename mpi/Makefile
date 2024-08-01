BASE = mpicxx -o mpi_cuda_matmul mpi_cuda_matmul.cu -O3 -march=native -mp=gpu -std=c++20 -use_fast_math -lcudart -L/usr/local/cuda/lib64 --diag_suppress declared_but_not_referenced -Xlinker -znoexecstack -Xcompiler "-fopenmp,-fpic" -Xnvlink "-suppress-stack-size-warning"
TARGET = mpiexec -n 4 -ppn 4 --cpu-bind=list:0-7:8-15:16-23:24-31 -d 1 ./affinity_gpu_polaris.sh ./mpi_cuda_matmul
all:
	$(BASE) -DVECTORIZE

run: all
	$(TARGET) 20 20 20 1

corr:
	@$(BASE)
	$(TARGET) 200 200 200 1
	@$(BASE) -DVECTORIZE
	$(TARGET) 200 200 200 1
	@$(BASE) -DUSEOPENMP
	$(TARGET) 200 200 200 1
	@$(BASE) -DUSEOPENMP -DVECTORIZE
	$(TARGET) 200 200 200 1

test: 
	@$(BASE)
	$(TARGET) 20000 20000 20000
	@$(BASE) -DVECTORIZE
	$(TARGET) 20000 20000 20000
	@$(BASE) -DUSEOPENMP
	$(TARGET) 20000 20000 20000
	@$(BASE) -DUSEOPENMP -DVECTORIZE
	$(TARGET) 20000 20000 20000

openmp:
	@$(BASE) -DUSEOPENMP
	$(TARGET) 20000 20000 20000
	@$(BASE) -DUSEOPENMP -DVECTORIZE
	$(TARGET) 20000 20000 20000