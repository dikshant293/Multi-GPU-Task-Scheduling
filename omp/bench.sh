#!/bin/bash

# Build the program using make
make

# Check if make succeeded
if [ $? -ne 0 ]; then
  echo "Make failed, exiting script."
  exit 1
fi

# Loop to run the program with g from 0.0 to 1.0 with increments of 0.05
for g in $(seq 0.0 0.05 1.0)
do
  echo "Running with g value: $g"
  ./t2g_llvm_nv 1000 1000 1000 $g
done