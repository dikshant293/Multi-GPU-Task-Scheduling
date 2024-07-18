all:
	mpicxx -o mpi_cuda_matmul mpi_cuda_matmul.cu -lcudart -L/usr/local/cuda/lib64

run: all
	mpiexec -n 4 ./mpi_cuda_matmul