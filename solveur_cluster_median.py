# ============================================================
# solveur_cluster_median.py
# ============================================================

from amplpy import AMPL
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import time


def plot_clusters(points, assign, k, obj):
    colors = plt.cm.tab10.colors
    plt.figure(figsize=(8, 6))

    for idx, c in enumerate(np.unique(assign)):
        pts = np.where(assign == c)[0]
        col = colors[idx % len(colors)]
        plt.scatter(points[pts, 0], points[pts, 1], color=col, s=60)
        plt.scatter(points[c, 0], points[c, 1],
                    marker="*", s=300, color=col,
                    edgecolor="black", linewidth=1.5)

    plt.title(f"AMPL optimal clustering (k={k})\nObjective = {obj:.3f}")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig("solveur_clusters.png", dpi=300)
    plt.close()


def solve_cluster_median(mod_file, dat_file):
    ampl = AMPL()
    ampl.read(mod_file)
    ampl.readData(dat_file)
    ampl.setOption("solver", "cplex")

    start = time.time()
    ampl.solve()
    cpu = time.time() - start

    m = int(ampl.getParameter("m").value())
    k = int(ampl.getParameter("k").value())
    x = ampl.get_data("x").toDict()
    obj = float(ampl.get_value("TotalDistance"))

    assign = np.zeros(m, dtype=int)
    for (i, j), v in x.items():
        if v > 0.5:
            assign[i-1] = j-1

    ampl.close()
    return obj, cpu, assign, k


if __name__ == "__main__":
    BASE = Path(__file__).resolve().parent

    points = np.load(BASE / "benchmark_points.npy")
    obj, cpu, assign, k = solve_cluster_median(
        "cluster_median.mod",
        "benchmark.dat"
    )

    print("AMPL objective:", obj)
    print("CPU time:", cpu)

    plot_clusters(points, assign, k, obj)
