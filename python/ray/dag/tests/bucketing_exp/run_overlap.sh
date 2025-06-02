#!/bin/bash
# filepath: run_bucket_sizes.sh

# Set the range of bucket sizes to test
BUCKET_SIZES=(10 20 50 100 200 500 1000)

# Set other parameters
NUM_COMPUTES=1000
ITERATIONS=5
OUTPUT_FILE="bucket_size_results.txt"

echo "Running benchmark with different bucket sizes..." > $OUTPUT_FILE
echo "----------------------------------------" >> $OUTPUT_FILE

for size in "${BUCKET_SIZES[@]}"; do
    echo "Running with bucket size: $size"
    
    # Run the Python script with the current bucket size
    python /home/wyao/ray/python/ray/dag/tests/ddp/overlap.py \
        --bucket-size $size \
        --num-computes $NUM_COMPUTES \
        --iterations $ITERATIONS >> $OUTPUT_FILE
    
    echo "----------------------------------------" >> $OUTPUT_FILE
    # Add a small delay between runs
    sleep 1
done

echo "Benchmark complete. Results saved in $OUTPUT_FILE"