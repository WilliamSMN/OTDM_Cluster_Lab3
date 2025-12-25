# ============================================================
# benchmark_cluster_median.py
# ============================================================

import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

from solveur_cluster_median import solve_cluster_median
from heuristic_cluster_median import (
    build_mst, cut_longest,
    extract_clusters, compute_objective
)


def main():
    BASE = Path(__file__).resolve().parent
    points = np.load(BASE / "benchmark_points.npy")
    k = 3

    # Exact
    obj_opt, cpu_opt, _, _ = solve_cluster_median(
        "cluster_median.mod",
        "benchmark.dat"
    )

    # Heuristic
    start = time.time()
    mst, D = build_mst(points)
    mst_cut, _ = cut_longest(mst, k)
    clusters = extract_clusters(mst_cut)
    obj_heur, _ = compute_objective(D, clusters)
    cpu_heur = time.time() - start

    gap = 100 * (obj_heur - obj_opt) / obj_opt

    df = pd.DataFrame([
        ["AMPL optimal", obj_opt, 0.0, cpu_opt],
        ["MST heuristic", obj_heur, gap, cpu_heur]
    ], columns=["Method", "Objective", "Gap (%)", "CPU time (s)"])

    print(df)

    plt.figure(figsize=(6, 4))
    plt.bar(df["Method"], df["Objective"])
    plt.ylabel("Cluster-median objective")
    plt.title("Objective comparison (same data)")
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig("benchmark_objective_comparison.png", dpi=300)
    plt.close()


if __name__ == "__main__":
    main()
