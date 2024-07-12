#===============================================================================
# Compiler Options
#===============================================================================

COMPILER    = nvhpc
OPTIMIZE    = yes
DEBUG       = no
PROFILE     = no
SM = cc70# --- NVIDIA arch
ARCH = gfx90a # --- AMD arch
ENABLE_ASYN = 0
SAVE_TEMP = 0

#===============================================================================
# Program name & source code list
#===============================================================================

OBJ = cudaMM.o
SRC = cudaMM.cu
TARGET = cuda_$(COMPILER)

#===============================================================================
# Sets Flags
#===============================================================================

# Standard Flags
CFLAGS := -std=c++20 -Wall -Xcompiler -fopenmp -march=native

# Linker Flags
LDFLAGS = -lm


# SCHFLAG = -DSCHED_ROUNDROBIN
SCHFLAG = -DSCHED_DYNAMIC
# SCHFLAG = -DSCHED_DYNAMIC2
# SCHFLAG = -DSCHED_RANDOM
# SCHFLAG = -DSCHED_ADAPTIVE
# SCHFLAG = -DSCHED_ADAPTIVE2

OPTFLAGS = -fast 
PRELOADFLAG = -DPRE_TRANSFER
VECFLAG = -DVECTORIZE
OMPFLAG = -DUSEOPENMP

# NVIDIA NVHPC Compiler 
ifeq ($(COMPILER),nvhpc)
  CC = nvc++
  #CFLAGS += -mp=gpu -gpu=managed
  CFLAGS += -mp=gpu -gpu=${SM}
  #CFLAGS += -mp=gpu -Minfo=mp -gpu=${SM}
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

CFLAGS += -I. $(OPTFLAGS)

CFLAGS += --diag_suppress declared_but_not_referenced --diag_suppress set_but_not_used

OBJS = $(OBJ)
TARGETS = $(TARGET)

BASE = $(CC) $(SRC) $(CFLAGS) -o $(TARGET) $(LDFLAGS)
NOVEC_NOPRE_CMP = $(BASE) $(SCHFLAG)
VEC_NOPRE_CMP = $(BASE) $(SCHFLAG) $(VECFLAG)
NOVEC_PRE_CMP = $(BASE) $(SCHFLAG)
VEC_PRE_CMP = $(BASE) $(SCHFLAG) $(VECFLAG) $(PRELOADFLAG)

all: clean $(TARGETS)

$(TARGET):
	$(CC) $(SRC) $(CFLAGS) -o $@ $+ $(LDFLAGS) $(PRELOADFLAG) $(SCHFLAG) $(VECFLAG)

.PHONY: clean

clean:
	rm -rf *~ *.dSYM nc.vg.* $(OBJS) $(TARGETS)

run: all
	./$(TARGET) 2000 2000 2000 0.9 256 64 1

seg: all
	./$(TARGET) 7 7 7 0.0 256 64 1

bench_size: all
	./$(TARGET) 200 200 200 0.99 256 64
	./$(TARGET) 2000 2000 2000 0.99 256 64
	./$(TARGET) 20000 20000 20000 0.99 256 64

bench_repeat: all
	./$(TARGET) 2000 2000 2000 0.99 256 64
	./$(TARGET) 2000 2000 2000 0.99 256 64
	./$(TARGET) 2000 2000 2000 0.99 256 64
	./$(TARGET) 2000 2000 2000 0.99 256 64
	./$(TARGET) 2000 2000 2000 0.99 256 64

n=20000
g=0.999
tpb=256
cmp:
	@$(NOVEC_NOPRE_CMP)
	./$(TARGET) $(n) $(n) $(n) $(g) $(tpb)
	@$(NOVEC_PRE_CMP)
	./$(TARGET) $(n) $(n) $(n) $(g) $(tpb)
	@$(VEC_NOPRE_CMP)
	./$(TARGET) $(n) $(n) $(n) $(g) $(tpb)
	@$(VEC_PRE_CMP)
	./$(TARGET) $(n) $(n) $(n) $(g) $(tpb)
	@rm -rf cmp $(TARGET)

omp:
	@$(BASE) $(PRELOADFLAG) $(VECFLAG) -DSCHED_ROUNDROBIN
	./$(TARGET) $(n) $(n) $(n) $(g) $(tpb)
	@$(BASE) $(PRELOADFLAG) $(VECFLAG) -DSCHED_ROUNDROBIN $(OMPFLAG)
	./$(TARGET) $(n) $(n) $(n) $(g) $(tpb)

# -DSCHED_ROUNDROBIN
# -DSCHED_DYNAMIC
# -DSCHED_DYNAMIC2
# -DSCHED_RANDOM
# -DSCHED_ADAPTIVE
# -DSCHED_ADAPTIVE2
sch:
	@$(BASE) $(PRELOADFLAG) $(VECFLAG) $(OMPFLAG) -DSCHED_ROUNDROBIN
	./$(TARGET) $(n) $(n) $(n) $(g) $(tpb)
	@$(BASE) $(PRELOADFLAG) $(VECFLAG) $(OMPFLAG) -DSCHED_RANDOM
	./$(TARGET) $(n) $(n) $(n) $(g) $(tpb)
	@$(BASE) $(PRELOADFLAG) $(VECFLAG) $(OMPFLAG) -DSCHED_DYNAMIC
	./$(TARGET) $(n) $(n) $(n) $(g) $(tpb)
	@$(BASE) $(PRELOADFLAG) $(VECFLAG) $(OMPFLAG) -DSCHED_DYNAMIC2
	./$(TARGET) $(n) $(n) $(n) $(g) $(tpb)
	@$(BASE) $(PRELOADFLAG) $(VECFLAG) $(OMPFLAG) -DSCHED_ADAPTIVE
	./$(TARGET) $(n) $(n) $(n) $(g) $(tpb)
	@$(BASE) $(PRELOADFLAG) $(VECFLAG) $(OMPFLAG) -DSCHED_ADAPTIVE2
	./$(TARGET) $(n) $(n) $(n) $(g) $(tpb)
	@rm -rf cmp $(TARGET)

ex:
	$(BASE) $(PRELOADFLAG) $(VECFLAG) -DSCHED_ADAPTIVE