BASE_CPP = mpicxx -o mpi_cuda_matmul mpi_cuda_matmul.cpp -O3 -march=native -mp=gpu -std=c++20 -use_fast_math --diag_suppress declared_but_not_referenced -Xlinker -znoexecstack -Xcompiler "-fopenmp,-fpic" -Xnvlink "-suppress-stack-size-warning"
BASE_CU = mpicxx -o mpi_cuda_matmul mpi_cuda_matmul.cu -O3 -march=native -mp=gpu -std=c++20 -use_fast_math --diag_suppress declared_but_not_referenced -Xlinker -znoexecstack -Xcompiler "-fopenmp,-fpic" -Xnvlink "-suppress-stack-size-warning"
TARGET = mpiexec -n 4 -ppn 4 --cpu-bind=list:0-7:8-15:16-23:24-31 -d 1 ./affinity_gpu_polaris.sh ./mpi_cuda_matmul
CUDA_LD = -lcudart -L/usr/local/cuda/lib64
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