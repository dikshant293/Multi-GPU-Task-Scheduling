BASE = mpicxx -o mpi_cuda_matmul mpi_cuda_matmul.cu -lcudart -L/usr/local/cuda/lib64 --diag_suppress declared_but_not_referenced
all:
	$(BASE) -DVECTORIZE

run: all
	mpiexec -n 4 ./mpi_cuda_matmul 20 20 20 1

corr:
	$(BASE)
	mpiexec -n 4 ./mpi_cuda_matmul 20 20 20 1
	$(BASE) -DVECTORIZE
	mpiexec -n 4 ./mpi_cuda_matmul 20 20 20 1

test: 
	$(BASE)
	mpiexec -n 4 ./mpi_cuda_matmul 20000 20000 20000
	$(BASE) -DVECTORIZE
	mpiexec -n 4 ./mpi_cuda_matmul 20000 20000 20000