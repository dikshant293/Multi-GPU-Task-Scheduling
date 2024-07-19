all:
	mpicxx -o mpi_cuda_matmul mpi_cuda_matmul.cu -lcudart -L/usr/local/cuda/lib64 --diag_suppress declared_but_not_referenced -DVECTORIZE

run: all
	mpiexec -n 4 ./mpi_cuda_matmul 20 20 20 1

corr:
	mpicxx -o mpi_cuda_matmul mpi_cuda_matmul.cu -lcudart -L/usr/local/cuda/lib64 --diag_suppress declared_but_not_referenced
	mpiexec -n 4 ./mpi_cuda_matmul 20 20 20 1
	mpicxx -o mpi_cuda_matmul mpi_cuda_matmul.cu -lcudart -L/usr/local/cuda/lib64 --diag_suppress declared_but_not_referenced -DVECTORIZE
	mpiexec -n 4 ./mpi_cuda_matmul 20 20 20 1

test: 
	mpicxx -o mpi_cuda_matmul mpi_cuda_matmul.cu -lcudart -L/usr/local/cuda/lib64 --diag_suppress declared_but_not_referenced
	mpiexec -n 4 ./mpi_cuda_matmul 20000 20000 20000
	mpicxx -o mpi_cuda_matmul mpi_cuda_matmul.cu -lcudart -L/usr/local/cuda/lib64 --diag_suppress declared_but_not_referenced -DVECTORIZE
	mpiexec -n 4 ./mpi_cuda_matmul 20000 20000 20000