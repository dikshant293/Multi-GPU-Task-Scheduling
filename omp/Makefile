#===============================================================================
# Compiler Options
#===============================================================================

COMPILER    = llvm_nv
OPTIMIZE    = yes
DEBUG       = no
PROFILE     = no
SM = cc70   # --- NVIDIA arch
ARCH = gfx90a # --- AMD arch
ENABLE_ASYN = 1
SAVE_TEMP = 0

#===============================================================================
# Program name & source code list
#===============================================================================

OBJ = ompMM.o
SRC = ompMM.cpp
TARGET = t2g_$(COMPILER)

#===============================================================================
# Sets Flags
#===============================================================================

# Standard Flags
CFLAGS := -std=c++20 -Wall

# Linker Flags
LDFLAGS = -lm

OPTFLAGS = -DSCHED_ROUNDROBIN

# GCC Compiler
ifeq ($(COMPILER),gnu)
  CC = gcc
  CFLAGS += -fopenmp -flto
endif

# Intel Compiler
ifeq ($(COMPILER),intel)
  CC = icx 
  CFLAGS += -fiopenmp -fopenmp-targets=spir64 -D__STRICT_ANSI__ 
endif

# LLVM Clang Compiler 
ifeq ($(COMPILER),llvm_nv)
  CC = clang++
  CFLAGS += -fopenmp -ffast-math -fopenmp-targets=nvptx64-nvidia-cuda  -fopenmp-cuda-mode -Wno-unused-variable -Wno-unused-but-set-variable -Wno-sometimes-uninitialized -Ofast -openmp-simd -openmp-extensions -foffload-lto -fopenmp-assume-no-thread-state -march=native -fno-stack-protector -funroll-loops -fomit-frame-pointer -mavx2 -ffp-contract=fast -fno-signed-zeros -freciprocal-math -fmerge-all-constants
  #CFLAGS += -fopenmp -fopenmp-targets=nvptx64-nvidia-cuda -Xopenmp-target --cuda-path=${OLCF_CUDA_ROOT}  -Xcuda-ptxas --maxrregcount=60 -fopenmp-cuda-mode
  #CFLAGS += -fopenmp -fopenmp-targets=nvptx64-nvidia-cuda -Xopenmp-target --cuda-path=${OLCF_CUDA_ROOT}    -fopenmp-new-driver -foffload-lto 
  #CFLAGS += -fopenmp -fopenmp-targets=nvptx64-nvidia-cuda -Xopenmp-target --cuda-path=${OLCF_CUDA_ROOT}  -Xcuda-ptxas --maxrregcount=120 -fopenmp-new-driver -foffload-lto -fopenmp-assume-no-thread-state
endif

# IBM XL Compiler
ifeq ($(COMPILER),xl)
  CC = xlC
  CFLAGS += -qsmp=omp -qoffload -qstrict
endif

# NVIDIA NVHPC Compiler 
ifeq ($(COMPILER),nvhpc)
  CC = nvc++
  #CFLAGS += -mp=gpu -gpu=managed
  CFLAGS += -mp=gpu -gpu=${SM} --diag_suppress declared_but_not_referenced --diag_suppress set_but_not_used
  #CFLAGS += -mp=gpu -Minfo=mp -gpu=${SM}
endif

# AOMP Compiler
ifeq ($(COMPILER),llvm_amd)
  CC = clang++
  CFLAGS += -fopenmp -fopenmp-targets=amdgcn-amd-amdhsa -Xopenmp-target=amdgcn-amd-amdhsa -march=${ARCH}
endif

# Debug Flags
ifeq ($(DEBUG),yes)
  CFLAGS += -g
  LDFLAGS  += -g
endif

# Profiling Flags
ifeq ($(PROFILE),yes)
  CFLAGS += -pg
  LDFLAGS  += -pg
endif

# Optimization Flags
ifeq ($(OPTIMIZE),yes)
  CFLAGS += -O3
endif

# Using device offload
ifeq ($(ENABLE_ASYN),1)
  CFLAGS += -DASYN
endif

# Compiler Trace  
ifeq ($(SAVE_TEMPS),1)
CFLAGS += -save-temps
endif


#===============================================================================
# Targets to Build
#===============================================================================

CFLAGS += -I.

OBJS = $(OBJ)
TARGETS = $(TARGET)

PRELOADFLAG = -DPRE_TRANSFER
BASE = $(CC) $(SRC) $(CFLAGS) -o $(TARGET) $(LDFLAGS)
NOVEC_NOPRE_CMP = $(BASE) $(SCHFLAG)
VEC_NOPRE_CMP = $(BASE) $(SCHFLAG) $(VECFLAG)
NOVEC_PRE_CMP = $(BASE) $(SCHFLAG)
VEC_PRE_CMP = $(BASE) $(SCHFLAG) $(VECFLAG) $(PRELOADFLAG)

all: clean
	$(BASE) -DSCHED_ROUNDROBIN

$(TARGET):  $(OBJ)
	$(CC) $(CFLAGS) -o $@ $+ $(LDFLAGS)

$(OBJ): $(SRC)
	$(CC) $(INCLUDE) $(CFLAGS) -c $< -o $@

.PHONY: clean

clean:
	rm -rf *~ *.dSYM nc.vg.* $(OBJS) $(TARGETS)

run: clean
	$(BASE) $(PRELOADFLAG) -DSCHED_ROUNDROBIN
	./$(TARGET) 20 20 20 0.9 0 1
	$(BASE) -DSCHED_ROUNDROBIN
	./$(TARGET) 20 20 20 0.9 0 1
	$(BASE) -DSCHED_ROUNDROBIN -DVECTORIZE
	./$(TARGET) 20 20 20 0.9 0 1

n=20000
g=0.99
tpb=256

load: clean
	@$(BASE) $(PRELOADFLAG) -DSCHED_ROUNDROBIN
	./$(TARGET) $(n) $(n) $(n) $(g)
	@$(BASE) -DSCHED_ROUNDROBIN
	./$(TARGET) $(n) $(n) $(n) $(g)

gran:
	@$(BASE) $(PRELOADFLAG) -DSCHED_ROUNDROBIN
	./$(TARGET) $(n) $(n) $(n) 0.0
	./$(TARGET) $(n) $(n) $(n) 0.1
	./$(TARGET) $(n) $(n) $(n) 0.5
	./$(TARGET) $(n) $(n) $(n) 0.9
	./$(TARGET) $(n) $(n) $(n) 0.99
	./$(TARGET) $(n) $(n) $(n) 0.999
	./$(TARGET) $(n) $(n) $(n) 0.9999
	./$(TARGET) $(n) $(n) $(n) 0.99999
	@rm -rf cmp $(TARGET)

sch-nopre:
	@$(BASE) $(PRELOADFLAG) -DSCHED_ROUNDROBIN
	./$(TARGET) $(n) $(n) $(n) $(g)
	@$(BASE) $(PRELOADFLAG) -DSCHED_RANDOM
	./$(TARGET) $(n) $(n) $(n) $(g)
	@$(BASE) $(PRELOADFLAG) -DSCHED_DYNAMIC2
	./$(TARGET) $(n) $(n) $(n) $(g)
	@$(BASE) $(PRELOADFLAG) -DSCHED_ADAPTIVE2
	./$(TARGET) $(n) $(n) $(n) $(g)
	@rm -rf cmp $(TARGET)

sch:
	@$(BASE) -DSCHED_ROUNDROBIN
	./$(TARGET) $(n) $(n) $(n) $(g)
	@$(BASE) -DSCHED_RANDOM
	./$(TARGET) $(n) $(n) $(n) $(g)
	@$(BASE) -DSCHED_DYNAMIC2
	./$(TARGET) $(n) $(n) $(n) $(g)
	@$(BASE) -DSCHED_ADAPTIVE2
	./$(TARGET) $(n) $(n) $(n) $(g)
	@rm -rf cmp $(TARGET)

vec:
	@$(BASE) -DSCHED_ROUNDROBIN
	./$(TARGET) $(n) $(n) $(n) $(g)
	@$(BASE) -DSCHED_ROUNDROBIN -DVECTORIZE
	./$(TARGET) $(n) $(n) $(n) $(g)


cmp-sch:
	@echo Equal
	@$(BASE) -DSCHED_ROUNDROBIN
	./$(TARGET) $(n) $(n) $(n) $(g)
	@$(BASE) -DSCHED_RANDOM
	./$(TARGET) $(n) $(n) $(n) $(g)
	@$(BASE) -DSCHED_DYNAMIC2
	./$(TARGET) $(n) $(n) $(n) $(g)
	@$(BASE) -DSCHED_ADAPTIVE2
	./$(TARGET) $(n) $(n) $(n) $(g)
	@echo
	@echo Uniform	
	@$(BASE) -DSCHED_ROUNDROBIN
	./$(TARGET) $(n) $(n) $(n) $(g) 1
	@$(BASE) -DSCHED_RANDOM
	./$(TARGET) $(n) $(n) $(n) $(g) 1
	@$(BASE) -DSCHED_DYNAMIC2
	./$(TARGET) $(n) $(n) $(n) $(g) 1
	@$(BASE) -DSCHED_ADAPTIVE2
	./$(TARGET) $(n) $(n) $(n) $(g) 1
	@echo
	@echo Random
	@$(BASE) -DSCHED_ROUNDROBIN
	./$(TARGET) $(n) $(n) $(n) $(g) 2
	@$(BASE) -DSCHED_RANDOM
	./$(TARGET) $(n) $(n) $(n) $(g) 2
	@$(BASE) -DSCHED_DYNAMIC2
	./$(TARGET) $(n) $(n) $(n) $(g) 2
	@$(BASE) -DSCHED_ADAPTIVE2
	./$(TARGET) $(n) $(n) $(n) $(g) 2


gran-test:
	@$(BASE) -DSCHED_ROUNDROBIN
	./$(TARGET) $(n) $(n) $(n) 0.0
	./$(TARGET) $(n) $(n) $(n) 0.1
	./$(TARGET) $(n) $(n) $(n) 0.4
	./$(TARGET) $(n) $(n) $(n) 0.6
	./$(TARGET) $(n) $(n) $(n) 0.8
	./$(TARGET) $(n) $(n) $(n) 0.9
	./$(TARGET) $(n) $(n) $(n) 0.95
	./$(TARGET) $(n) $(n) $(n) 0.99
	./$(TARGET) $(n) $(n) $(n) 0.995
	./$(TARGET) $(n) $(n) $(n) 0.999
	./$(TARGET) $(n) $(n) $(n) 0.9999
	./$(TARGET) $(n) $(n) $(n) 1.0

	@$(BASE) -DSCHED_RANDOM
	./$(TARGET) $(n) $(n) $(n) 0.0
	./$(TARGET) $(n) $(n) $(n) 0.1
	./$(TARGET) $(n) $(n) $(n) 0.4
	./$(TARGET) $(n) $(n) $(n) 0.6
	./$(TARGET) $(n) $(n) $(n) 0.8
	./$(TARGET) $(n) $(n) $(n) 0.9
	./$(TARGET) $(n) $(n) $(n) 0.95
	./$(TARGET) $(n) $(n) $(n) 0.99
	./$(TARGET) $(n) $(n) $(n) 0.995
	./$(TARGET) $(n) $(n) $(n) 0.999
	./$(TARGET) $(n) $(n) $(n) 0.9999
	./$(TARGET) $(n) $(n) $(n) 1.0

	@$(BASE) -DSCHED_DYNAMIC2
	./$(TARGET) $(n) $(n) $(n) 0.0
	./$(TARGET) $(n) $(n) $(n) 0.1
	./$(TARGET) $(n) $(n) $(n) 0.4
	./$(TARGET) $(n) $(n) $(n) 0.6
	./$(TARGET) $(n) $(n) $(n) 0.8
	./$(TARGET) $(n) $(n) $(n) 0.9
	./$(TARGET) $(n) $(n) $(n) 0.95
	./$(TARGET) $(n) $(n) $(n) 0.99
	./$(TARGET) $(n) $(n) $(n) 0.995
	./$(TARGET) $(n) $(n) $(n) 0.999
	./$(TARGET) $(n) $(n) $(n) 0.9999
	./$(TARGET) $(n) $(n) $(n) 1.0

	@$(BASE) -DSCHED_ADAPTIVE2
	./$(TARGET) $(n) $(n) $(n) 0.0
	./$(TARGET) $(n) $(n) $(n) 0.1
	./$(TARGET) $(n) $(n) $(n) 0.4
	./$(TARGET) $(n) $(n) $(n) 0.6
	./$(TARGET) $(n) $(n) $(n) 0.8
	./$(TARGET) $(n) $(n) $(n) 0.9
	./$(TARGET) $(n) $(n) $(n) 0.95
	./$(TARGET) $(n) $(n) $(n) 0.99
	./$(TARGET) $(n) $(n) $(n) 0.995
	./$(TARGET) $(n) $(n) $(n) 0.999
	./$(TARGET) $(n) $(n) $(n) 0.9999
	./$(TARGET) $(n) $(n) $(n) 1.0

gran-7:
	@$(BASE) -DSCHED_ROUNDROBIN
	./$(TARGET) $(n) $(n) $(n) 0.7

	@$(BASE) -DSCHED_RANDOM
	./$(TARGET) $(n) $(n) $(n) 0.7

	@$(BASE) -DSCHED_DYNAMIC2
	./$(TARGET) $(n) $(n) $(n) 0.7

	@$(BASE) -DSCHED_ADAPTIVE2
	./$(TARGET) $(n) $(n) $(n) 0.7
	
