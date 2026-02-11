import pandas as pd
import matplotlib.pyplot as plt

file_path = "benchmark_cpu_vs_cuda.csv"
df = pd.read_csv(file_path)


df["speedup"] = df["cpu_avg_time"] / df["cuda_avg_time"]

grouped = df.groupby("dimension")["speedup"].mean().reset_index()

plt.figure()
plt.plot(grouped["dimension"], grouped["speedup"])
plt.xlabel("Dimension")
plt.ylabel("Average Speedup (CPU / CUDA)")
plt.title("CUDA Speedup over Parallel CPU")
plt.grid(True)
plt.savefig('speedup.png') 

