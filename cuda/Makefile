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
CFLAGS := -std=c++20 -Wall

# Linker Flags
LDFLAGS = -lm


# SCHFLAG = -DSCHED_DYNAMIC2
# SCHFLAG = -DSCHED_RANDOM

OPTFLAGS = -fast 
PRELOADFLAG = -DPRE_TRANSFER
VECFLAG = -DVECTORIZE
OMPFLAG = -DUSEOPENMP

# NVIDIA NVHPC Compiler 
ifeq ($(COMPILER),nvhpc)
  CC = nvc++
  #CFLAGS += -mp=gpu -gpu=managed
  CFLAGS += -O3 -I. -fast -mp=gpu -gpu=${SM} -Xcompiler "-fopenmp,-fpic" -march=native -Xlinker -znoexecstack
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
	./$(TARGET) 2000 2000 2000 0.99 0 1
	

n=20000
g=0.999
tpb=1024
strm=4

cmp:
	@$(NOVEC_NOPRE_CMP)
	./$(TARGET) $(n) $(n) $(n) $(g)
	@$(NOVEC_PRE_CMP)
	./$(TARGET) $(n) $(n) $(n) $(g)
	@$(VEC_NOPRE_CMP)
	./$(TARGET) $(n) $(n) $(n) $(g)
	@$(VEC_PRE_CMP)
	./$(TARGET) $(n) $(n) $(n) $(g)
	@rm -rf cmp $(TARGET)

omp:
	@$(BASE) $(PRELOADFLAG) $(VECFLAG) -DSCHED_ROUNDROBIN
	./$(TARGET) $(n) $(n) $(n) $(g)
	@$(BASE) $(PRELOADFLAG) $(VECFLAG) -DSCHED_ROUNDROBIN $(OMPFLAG)
	./$(TARGET) $(n) $(n) $(n) $(g)

# -DSCHED_DYNAMIC2
# -DSCHED_RANDOM
sch:
	@$(BASE) $(PRELOADFLAG) $(VECFLAG) $(OMPFLAG) -DSCHED_ROUNDROBIN
	./$(TARGET) $(n) $(n) $(n) $(g)
	@$(BASE) $(PRELOADFLAG) $(VECFLAG) $(OMPFLAG) -DSCHED_RANDOM
	./$(TARGET) $(n) $(n) $(n) $(g)
	@$(BASE) $(PRELOADFLAG) $(VECFLAG) $(OMPFLAG) -DSCHED_DYNAMIC2
	./$(TARGET) $(n) $(n) $(n) $(g)
	@$(BASE) $(PRELOADFLAG) $(VECFLAG) $(OMPFLAG) -DSCHED_ADAPTIVE2
	./$(TARGET) $(n) $(n) $(n) $(g)
	@rm -rf cmp $(TARGET)

	
gran:
	@$(BASE) $(PRELOADFLAG) $(VECFLAG) $(OMPFLAG) -DSCHED_ROUNDROBIN
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
	@$(BASE) $(VECFLAG) $(OMPFLAG) -DSCHED_ROUNDROBIN
	./$(TARGET) $(n) $(n) $(n) $(g)
	@$(BASE) $(VECFLAG) $(OMPFLAG) -DSCHED_RANDOM
	./$(TARGET) $(n) $(n) $(n) $(g)
	@$(BASE) $(VECFLAG) $(OMPFLAG) -DSCHED_DYNAMIC2
	./$(TARGET) $(n) $(n) $(n) $(g)
	@$(BASE) $(VECFLAG) $(OMPFLAG) -DSCHED_ADAPTIVE2
	./$(TARGET) $(n) $(n) $(n) $(g)
	@rm -rf cmp $(TARGET)

ex:
	$(BASE) $(PRELOADFLAG) $(VECFLAG) -DSCHED_ADAPTIVE

# streams: clean
# 	@$(BASE) $(PRELOADFLAG) $(VECFLAG) -DSCHED_ROUNDROBIN $(OMPFLAG)
# 	./$(TARGET) $(n) $(n) $(n) $(g) $(tpb) 1
# 	./$(TARGET) $(n) $(n) $(n) $(g) $(tpb) 2
# 	./$(TARGET) $(n) $(n) $(n) $(g) $(tpb) 4
# 	./$(TARGET) $(n) $(n) $(n) $(g) $(tpb) 6
# 	./$(TARGET) $(n) $(n) $(n) $(g) $(tpb) 8
# 	./$(TARGET) $(n) $(n) $(n) $(g) $(tpb) 12
# 	./$(TARGET) $(n) $(n) $(n) $(g) $(tpb) 16
# 	./$(TARGET) $(n) $(n) $(n) $(g) $(tpb) 24
# 	./$(TARGET) $(n) $(n) $(n) $(g) $(tpb) 32
# 	./$(TARGET) $(n) $(n) $(n) $(g) $(tpb) 48
# 	./$(TARGET) $(n) $(n) $(n) $(g) $(tpb) 64


cmp-sch:
	@echo Equal
	@$(BASE) $(PRELOADFLAG) $(VECFLAG) $(OMPFLAG) -DSCHED_ROUNDROBIN
	./$(TARGET) $(n) $(n) $(n) $(g)
	@$(BASE) $(PRELOADFLAG) $(VECFLAG) $(OMPFLAG) -DSCHED_RANDOM
	./$(TARGET) $(n) $(n) $(n) $(g)
	@$(BASE) $(PRELOADFLAG) $(VECFLAG) $(OMPFLAG) -DSCHED_DYNAMIC2
	./$(TARGET) $(n) $(n) $(n) $(g)
	@$(BASE) $(PRELOADFLAG) $(VECFLAG) $(OMPFLAG) -DSCHED_ADAPTIVE2
	./$(TARGET) $(n) $(n) $(n) $(g)
	@echo
	@echo Uniform	
	@$(BASE) $(PRELOADFLAG) $(VECFLAG) $(OMPFLAG) -DSCHED_ROUNDROBIN
	./$(TARGET) $(n) $(n) $(n) $(g) 1
	@$(BASE) $(PRELOADFLAG) $(VECFLAG) $(OMPFLAG) -DSCHED_RANDOM
	./$(TARGET) $(n) $(n) $(n) $(g) 1
	@$(BASE) $(PRELOADFLAG) $(VECFLAG) $(OMPFLAG) -DSCHED_DYNAMIC2
	./$(TARGET) $(n) $(n) $(n) $(g) 1
	@$(BASE) $(PRELOADFLAG) $(VECFLAG) $(OMPFLAG) -DSCHED_ADAPTIVE2
	./$(TARGET) $(n) $(n) $(n) $(g) 1
	@echo
	@echo Random
	@$(BASE) $(PRELOADFLAG) $(VECFLAG) $(OMPFLAG) -DSCHED_ROUNDROBIN
	./$(TARGET) $(n) $(n) $(n) $(g) 2
	@$(BASE) $(PRELOADFLAG) $(VECFLAG) $(OMPFLAG) -DSCHED_RANDOM
	./$(TARGET) $(n) $(n) $(n) $(g) 2
	@$(BASE) $(PRELOADFLAG) $(VECFLAG) $(OMPFLAG) -DSCHED_DYNAMIC2
	./$(TARGET) $(n) $(n) $(n) $(g) 2
	@$(BASE) $(PRELOADFLAG) $(VECFLAG) $(OMPFLAG) -DSCHED_ADAPTIVE2
	./$(TARGET) $(n) $(n) $(n) $(g) 2


info:
	$(BASE) $(PRELOADFLAG) $(VECFLAG) $(OMPFLAG) -DSCHED_ROUNDROBIN -Minfo