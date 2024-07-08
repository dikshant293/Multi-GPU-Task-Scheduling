#!/bin/bash

# Build the project
make

# Check if make was successful
if [ $? -ne 0 ]; then
    echo "Make failed, exiting."
    exit 1
fi

# Initial value for t
t=1

# Loop until t exceeds 1024
while [ $t -le 1024 ]; do
    # Run the program with current value of t
    echo "Running with t=$t"
    ./cuda_nvhpc $1 $1 $1 0.9 $t
    
    # Double the value of t
    t=$((t * 2))
done

echo "Script completed."
