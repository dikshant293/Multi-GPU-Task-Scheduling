#===============================================================================
# User Options
#===============================================================================

# Compiler can be set below, or via environment variable
CC        = clang++
OPTIMIZE  = yes
DEBUG     = no
DEVICE    = gpu
ARCH      = gfx906

#===============================================================================
# Program name & source code list
#===============================================================================

TARGET = motionsim_$(CC)

SRC = motionsim.cpp

obj = $(source:.cpp=.o)

#===============================================================================
# Sets Flags
#===============================================================================

# Standard Flags
CFLAGS := -std=c++20  -Wall 

# Linker Flags
LDFLAGS = 

# Debug Flags
ifeq ($(DEBUG),yes)
  CFLAGS += -g
  LDFLAGS  += -g
endif


# Optimization Flags
ifeq ($(OPTIMIZE),yes)
  CFLAGS += -O3
endif

ifeq ($(DEVICE),gpu)
  CFLAGS += -fopenmp -ffast-math -fopenmp-targets=nvptx64-nvidia-cuda  -fopenmp-cuda-mode -Wno-unused-variable -Wno-unused-but-set-variable -Wno-sometimes-uninitialized -Ofast -openmp-simd -openmp-extensions -foffload-lto -fopenmp-assume-no-thread-state -march=native -fno-stack-protector -funroll-loops -fomit-frame-pointer -mavx2 -ffp-contract=fast -fno-signed-zeros -freciprocal-math -fmerge-all-constants
else
  CFLAGS +=-fopenmp
endif
#===============================================================================
# Targets to Build
#===============================================================================

BASE = $(CC) $(SRC) $(CFLAGS) -o $(TARGET) $(LDFLAGS)

all: clean
	$(BASE) -DSCHED_ROUNDROBIN

clean:
	rm -rf $(TARGET) $(obj)

run: clean all
	./$(TARGET) 2000 0.9

n=20000
m=20000

test: clean all
	./$(TARGET) $(m) 0.0 $(n)
	./$(TARGET) $(m) 0.5 $(n)
	./$(TARGET) $(m) 0.75 $(n)
	./$(TARGET) $(m) 0.9 $(n)
	./$(TARGET) $(m) 0.99 $(n)
	./$(TARGET) $(m) 0.999 $(n)
	./$(TARGET) $(m) 0.9999 $(n)


gran-test:
	@$(BASE) -DSCHED_ROUNDROBIN
	./$(TARGET) $(m) 0.0 $(n)
	./$(TARGET) $(m) 0.1 $(n)
	./$(TARGET) $(m) 0.6 $(n)
	./$(TARGET) $(m) 0.7 $(n)
	./$(TARGET) $(m) 0.8 $(n)
	./$(TARGET) $(m) 0.9 $(n)
	./$(TARGET) $(m) 0.95 $(n)
	./$(TARGET) $(m) 0.99 $(n)
	./$(TARGET) $(m) 0.995 $(n)
	./$(TARGET) $(m) 0.999 $(n)
	./$(TARGET) $(m) 0.9999 $(n)
	./$(TARGET) $(m) 1.0 $(n)

	@$(BASE) -DSCHED_RANDOM
	./$(TARGET) $(m) 0.0 $(n)
	./$(TARGET) $(m) 0.1 $(n)
	./$(TARGET) $(m) 0.6 $(n)
	./$(TARGET) $(m) 0.7 $(n)
	./$(TARGET) $(m) 0.8 $(n)
	./$(TARGET) $(m) 0.9 $(n)
	./$(TARGET) $(m) 0.95 $(n)
	./$(TARGET) $(m) 0.99 $(n)
	./$(TARGET) $(m) 0.995 $(n)
	./$(TARGET) $(m) 0.999 $(n)
	./$(TARGET) $(m) 0.9999 $(n)
	./$(TARGET) $(m) 1.0 $(n)

	@$(BASE) -DSCHED_DYNAMIC2
	./$(TARGET) $(m) 0.0 $(n)
	./$(TARGET) $(m) 0.1 $(n)
	./$(TARGET) $(m) 0.6 $(n)
	./$(TARGET) $(m) 0.7 $(n)
	./$(TARGET) $(m) 0.8 $(n)
	./$(TARGET) $(m) 0.9 $(n)
	./$(TARGET) $(m) 0.95 $(n)
	./$(TARGET) $(m) 0.99 $(n)
	./$(TARGET) $(m) 0.995 $(n)
	./$(TARGET) $(m) 0.999 $(n)
	./$(TARGET) $(m) 0.9999 $(n)
	./$(TARGET) $(m) 1.0 $(n)

	@$(BASE) -DSCHED_ADAPTIVE2
	./$(TARGET) $(m) 0.0 $(n)
	./$(TARGET) $(m) 0.1 $(n)
	./$(TARGET) $(m) 0.6 $(n)
	./$(TARGET) $(m) 0.7 $(n)
	./$(TARGET) $(m) 0.8 $(n)
	./$(TARGET) $(m) 0.9 $(n)
	./$(TARGET) $(m) 0.95 $(n)
	./$(TARGET) $(m) 0.99 $(n)
	./$(TARGET) $(m) 0.995 $(n)
	./$(TARGET) $(m) 0.999 $(n)
	./$(TARGET) $(m) 0.9999 $(n)
	./$(TARGET) $(m) 1.0 $(n)

dyn:
	@$(BASE) -DSCHED_DYNAMIC2
	./$(TARGET) $(m) 0.0 $(n)
	./$(TARGET) $(m) 0.1 $(n)
	./$(TARGET) $(m) 0.6 $(n)
	./$(TARGET) $(m) 0.7 $(n)
	./$(TARGET) $(m) 0.8 $(n)
	./$(TARGET) $(m) 0.9 $(n)
	./$(TARGET) $(m) 0.95 $(n)
	./$(TARGET) $(m) 0.99 $(n)
	./$(TARGET) $(m) 0.995 $(n)
	./$(TARGET) $(m) 0.999 $(n)
	./$(TARGET) $(m) 0.9999 $(n)
	./$(TARGET) $(m) 1.0 $(n)