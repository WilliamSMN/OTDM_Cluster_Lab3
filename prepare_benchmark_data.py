# ============================================================
# prepare_benchmark_data.py
# Generates:
#  - benchmark_points.npy
#  - benchmark.dat
# ============================================================

import numpy as np
from pathlib import Path
from scipy.io import arff as arff_io


def pairwise_distances(points):
    diff = points[:, None, :] - points[None, :, :]
    return np.sqrt((diff ** 2).sum(axis=2))


BASE = Path(__file__).resolve().parent
ARFF_FILE = BASE / "data_processing/datasets/2d-3c-no123.arff"

# -------- CONFIG --------
SAMPLE_RATIO = 0.15   
SEED = 42
K = 5
# ------------------------

data, _ = arff_io.loadarff(ARFF_FILE)
points_all = np.array([[float(r[0]), float(r[1])] for r in data])

rng = np.random.default_rng(SEED)
m = int(round(SAMPLE_RATIO * len(points_all)))
idx = np.sort(rng.choice(len(points_all), size=m, replace=False))

points = points_all[idx]
D = pairwise_distances(points)

# Save points
np.save(BASE / "benchmark_points.npy", points)

# Write AMPL .dat
lines = []
lines.append("# Benchmark data for cluster-median\n")
lines.append(f"# Sample ratio: {SAMPLE_RATIO}, seed={SEED}\n")
lines.append(f"param m := {m};\n")
lines.append(f"param k := {K};\n\n")
lines.append("param d :\n")
lines.append("    " + " ".join(str(j+1) for j in range(m)) + " :=\n")
for i in range(m):
    lines.append(f"{i+1} " + " ".join(f"{D[i,j]:.6f}" for j in range(m)) + "\n")
lines.append(";\n")

(Path(BASE / "benchmark.dat")).write_text("".join(lines))

print("Generated benchmark.dat and benchmark_points.npy")
print(f"m = {m}, k = {K}")

