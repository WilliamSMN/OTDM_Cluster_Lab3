import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

from data_processing.prepare_benchmark_data import generate_benchmark_data
from helper.solveur_cluster_median import solve_cluster_median, plot_clusters as plot_opt
from helper.heuristic_cluster_median import (
    build_mst, cut_longest,
    extract_clusters, compute_objective,
    plot_clusters as plot_heur
)


def main():
    BASE = Path(__file__).resolve().parent

    # BENCHMARK PARAMETERS
    k = 13 #depends on the dataset used
    sample_ratio = 0.25
    seed = 42

    print("- Generate data")
    points, D, m = generate_benchmark_data(
        sample_ratio=sample_ratio,
        k=k,
        seed=seed,
        arff_file= "./data_processing/datasets/arrhythmia.arff",
        base_dir=BASE
    )

    print("- Solving exact cluster median using AMPL")
    obj_opt, cpu_opt, assign_opt, k_opt = solve_cluster_median(
        "cluster_median.mod",
        "benchmark.dat"
    )

    # Safety check (important)
    assert k_opt == k, "Mismatch between benchmark k and AMPL k!"

    plot_opt(
        points,
        assign_opt,
        k_opt,
        obj_opt,
        filename="benchmark_optimal_clusters.png"
    )

    print("- Computing MST heuristic")
    start = time.time()
    mst, D = build_mst(points)
    mst_cut, _ = cut_longest(mst, k)
    clusters = extract_clusters(mst_cut)
    obj_heur, assign_heur = compute_objective(D, clusters)
    cpu_heur = time.time() - start

    plot_heur(
        points,
        assign_heur,
        k,
        obj_heur,
        filename="benchmark_heuristic_clusters.png"
    )

    # Comparison table
    gap = 100 * (obj_heur - obj_opt) / obj_opt

    df = pd.DataFrame([
        ["AMPL optimal", obj_opt, 0.0, cpu_opt],
        ["MST heuristic", obj_heur, gap, cpu_heur]
    ], columns=["Method", "Objective", "Gap (%)", "CPU time (s)"])

    print(df)

    # Objective bar plot
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
