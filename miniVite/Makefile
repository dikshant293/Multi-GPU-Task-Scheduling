CXX = nvcc

# Set CUDA flags
CUDAFLAGS = -arch=sm_80 -O3 -Xcompiler -fopenmp -diag-suppress 177

# Compiler flags
CXXFLAGS = -std=c++11 $(CUDAFLAGS) -DUSE_32_BIT_GRAPH

TARGET = main_gpu

SRC = main.cu
OBJ = main.o

all: clean $(TARGET)

$(TARGET): $(OBJ)
	$(CXX) $(CXXFLAGS) $^ -o $@

%.o: %.cu
	$(CXX) $(CXXFLAGS) -c $< -o $@

run: $(TARGET)
	./$(TARGET) $(ARGS)

.PHONY: clean

clean:
	rm -rf *~ $(OBJ) $(TARGET)
