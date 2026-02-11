#!/bin/bash

output_file="benchmark_cpu_vs_cuda.csv"
echo "dimension,seed,threads,cpu_avg_time,cuda_avg_time" > "$output_file"

dims=(128 256 512 1024 1536 2048 3072 4096 6144 8192 16384)
threads=(20)
seeds=($RANDOM $RANDOM $RANDOM)

for dim in "${dims[@]}"; do
  for seed in "${seeds[@]}"; do
    for thread in "${threads[@]}"; do

      echo "==============================================="
      echo "Running benchmark: dim=$dim, seed=$seed, threads=$thread"
      echo "==============================================="
      echo "Running Parallel CPU..."

      cpu_output=$(PARLAY_NUM_THREADS="$thread" \
        ./build/benchmark_world_gen -dim "$dim" -seed "$seed" -p 2>&1)

      if echo "$cpu_output" | grep -q "FAILED"; then
        echo "CPU correctness check failed for dim=$dim, seed=$seed"
        exit 1
      fi

      cpu_avg=$(echo "$cpu_output" | grep "average" | awk '{print $2}')

      echo "Running CUDA..."

      cuda_output=$(./build/benchmark_world_gen -dim "$dim" -seed "$seed" -c 2>&1)

      if echo "$cuda_output" | grep -q "FAILED"; then
        echo "CUDA correctness check failed for dim=$dim, seed=$seed"
        exit 1
      fi

      cuda_avg=$(echo "$cuda_output" | grep "average" | awk '{print $2}')
        echo "$dim,$seed,$thread,$cpu_avg,$cuda_avg" >> "$output_file"

    done
  done
done

echo "Benchmarks completed. Results saved to $output_file"
