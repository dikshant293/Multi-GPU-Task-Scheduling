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
  CFLAGS += -fopenmp -ffast-math -fopenmp-targets=nvptx64-nvidia-cuda  -fopenmp-cuda-mode -Wno-unused-variable -Wno-unused-but-set-variable -Wno-sometimes-uninitialized
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
	$(BASE) $(PRELOADFLAG) -DSCHED_ROUNDROBIN

$(TARGET):  $(OBJ)
	$(CC) $(CFLAGS) -o $@ $+ $(LDFLAGS)

$(OBJ): $(SRC)
	$(CC) $(INCLUDE) $(CFLAGS) -c $< -o $@

.PHONY: clean

clean:
	rm -rf *~ *.dSYM nc.vg.* $(OBJS) $(TARGETS)

run: clean
	$(BASE) $(PRELOADFLAG) -DSCHED_ROUNDROBIN
	./$(TARGET) 2000 2000 2000 0.99 256 64 1
	$(BASE) -DSCHED_ROUNDROBIN
	./$(TARGET) 2000 2000 2000 0.99 256 64 1

n=20000
g=0.999
tpb=256

load: clean
	@$(BASE) $(PRELOADFLAG) -DSCHED_ROUNDROBIN
	./$(TARGET) $(n) $(n) $(n) $(g) $(tpb)
	@$(BASE) -DSCHED_ROUNDROBIN
	./$(TARGET) $(n) $(n) $(n) $(g) $(tpb)

gran:
	@$(BASE) $(PRELOADFLAG) -DSCHED_ROUNDROBIN
	./$(TARGET) $(n) $(n) $(n) 0.0 $(tpb) $(strm)
	./$(TARGET) $(n) $(n) $(n) 0.1 $(tpb) $(strm)
	./$(TARGET) $(n) $(n) $(n) 0.5 $(tpb) $(strm)
	./$(TARGET) $(n) $(n) $(n) 0.9 $(tpb) $(strm)
	./$(TARGET) $(n) $(n) $(n) 0.99 $(tpb) $(strm)
	./$(TARGET) $(n) $(n) $(n) 0.999 $(tpb) $(strm)
	./$(TARGET) $(n) $(n) $(n) 0.9999 $(tpb) $(strm)
	./$(TARGET) $(n) $(n) $(n) 0.99999 $(tpb) $(strm)
	@rm -rf cmp $(TARGET)

sch:
	@$(BASE) $(PRELOADFLAG) -DSCHED_ROUNDROBIN
	./$(TARGET) $(n) $(n) $(n) $(g) $(tpb)
	@$(BASE) $(PRELOADFLAG) -DSCHED_RANDOM
	./$(TARGET) $(n) $(n) $(n) $(g) $(tpb)
	@$(BASE) $(PRELOADFLAG) -DSCHED_DYNAMIC
	./$(TARGET) $(n) $(n) $(n) $(g) $(tpb)
	@$(BASE) $(PRELOADFLAG) -DSCHED_DYNAMIC2
	./$(TARGET) $(n) $(n) $(n) $(g) $(tpb)
	@$(BASE) $(PRELOADFLAG) -DSCHED_ADAPTIVE
	./$(TARGET) $(n) $(n) $(n) $(g) $(tpb)
	@$(BASE) $(PRELOADFLAG) -DSCHED_ADAPTIVE2
	./$(TARGET) $(n) $(n) $(n) $(g) $(tpb)
	@rm -rf cmp $(TARGET)

sch-nopre:
	@$(BASE) -DSCHED_ROUNDROBIN
	./$(TARGET) $(n) $(n) $(n) $(g) $(tpb)
	@$(BASE) -DSCHED_RANDOM
	./$(TARGET) $(n) $(n) $(n) $(g) $(tpb)
	@$(BASE) -DSCHED_DYNAMIC
	./$(TARGET) $(n) $(n) $(n) $(g) $(tpb)
	@$(BASE) -DSCHED_DYNAMIC2
	./$(TARGET) $(n) $(n) $(n) $(g) $(tpb)
	@$(BASE) -DSCHED_ADAPTIVE
	./$(TARGET) $(n) $(n) $(n) $(g) $(tpb)
	@$(BASE) -DSCHED_ADAPTIVE2
	./$(TARGET) $(n) $(n) $(n) $(g) $(tpb)
	@rm -rf cmp $(TARGET)