#!/bin/bash

output_file="benchmark_results.csv"
echo "dimension,seed,threads,heightmap_avg_time,tree_avg_time" > "$output_file"

# Dimensions to test
dims=(1024 2048 4096 8192)

# Thread counts to test
threads=(20)

# Generate 3 random seeds
seeds=($RANDOM $RANDOM $RANDOM)

for dim in "${dims[@]}"; do
  for seed in "${seeds[@]}"; do
    for thread in "${threads[@]}"; do
      echo "Running benchmark: dim=$dim, seed=$seed, threads=$thread"

      if [ "$thread" -eq 1 ]; then
        output=$(./build/benchmark_world_gen -dim "$dim" -seed "$seed" -s 2>&1)
      else
        output=$(PARLAY_NUM_THREADS="$thread" ./build/benchmark_world_gen -dim "$dim" -seed "$seed" -p 2>&1)
        if echo "$output" | grep -q "FAILED"; then
          echo "Correctness check failed for dim=$dim, seed=$seed, threads=$thread"
          exit 1
        fi
      fi
      heightmap_avg=$(echo "$output" | grep "average" | head -1 | awk '{print $2}')
      tree_avg=$(echo "$output" | grep "average" | tail -1 | awk '{print $2}')
      echo "$dim,$seed,$thread,$heightmap_avg,$tree_avg" >> "$output_file"
    done
  done
done

echo "Benchmarks completed. Results saved to $output_file"
