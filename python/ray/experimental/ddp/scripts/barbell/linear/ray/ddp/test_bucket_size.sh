#!/bin/bash

if [[ "$(pwd)" != */python/ray/experimental/ddp ]]; then
	echo "Please run in the python/ray/experimental/ddp directory"
	exit 1
fi

RED='\033[0;31m'
GREEN='\033[0;32m'
NC='\033[0m'

debug=false
while getopts "d" opt; do
	case $opt in
	d) debug=true ;;
	*)
		echo "Usage: $0 [-d]" >&2
		echo "  -d    Enable debug mode"
		exit 1
		;;
	esac
done

export TZ="America/Los_Angeles"
timestamp=$(date '+%Y%m%d_%H%M%S')

export RAY_DEDUP_LOGS=0

output_path=results/barbell/linear/ray/ddp/test_bucket_size
mkdir -p $output_path
rm -f $output_path/*.csv
rm -f $output_path/*.log
rm -f $output_path/*.png

layer_size=1280
num_layers=40
num_partitions_values=(
    1 2 5 10 20 40
)
num_actors=2
num_iters=10
latency_prefix=${timestamp}_ls${layer_size}_nl${num_layers}
model_prefix=$output_path/${timestamp}_model
log_file=$output_path/${timestamp}.log


# iterate over num_partitions_values
for i in "${!num_partitions_values[@]}"; do
    num_partitions="${num_partitions_values[$i]}"
    
    latency_prefix=${timestamp}_ls${layer_size}_nl${num_layers}_np${num_partitions}
    model_prefix=$output_path/${latency_prefix}_model
    log_file=${output_path}/${latency_prefix}.log

    echo "Running layer_size $layer_size, num_layers $num_layers, num_partitions $num_partitions..."
    python -m ray.experimental.ddp.src.main.linear.ray.ddp \
        --layer-size $layer_size \
        --num-layers $num_layers \
        --num-partitions $num_partitions \
        --num-actors $num_actors \
        --num-iters $num_iters \
        --output-path $output_path \
        --latency-prefix $latency_prefix \
        --save-model \
        --model-prefix $model_prefix \
        --tracing \
        >$log_file 2>&1
    status=$?
done

python utils/plot.py \
    --folder-path $output_path \
    --layer-size $layer_size \
    --num-layers $num_layers \

if $debug; then
	code $output_path/${timestamp}.log
fi

if [ $status -ne 0 ]; then
	echo -e "${RED}ER${NC}"
	exit 1
fi

echo -e "${GREEN}AC${NC}"
