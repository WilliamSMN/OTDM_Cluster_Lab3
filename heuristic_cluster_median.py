# ============================================================
# heuristic_cluster_median.py
# ============================================================

import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist, squareform
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import minimum_spanning_tree
from collections import deque
from pathlib import Path
import time


def build_mst(points):
    D = squareform(pdist(points))
    mst = minimum_spanning_tree(csr_matrix(D)).toarray()
    return mst + mst.T, D


def cut_longest(mst, k):
    edges = [(i, j, mst[i, j])
             for i in range(len(mst))
             for j in range(i+1, len(mst))
             if mst[i, j] > 0]
    edges.sort(key=lambda x: x[2], reverse=True)

    mst_cut = mst.copy()
    removed = edges[:k-1]
    for i, j, _ in removed:
        mst_cut[i, j] = mst_cut[j, i] = 0
    return mst_cut, removed


def extract_clusters(mst_cut):
    n = mst_cut.shape[0]
    visited = [False]*n
    clusters = []

    for i in range(n):
        if not visited[i]:
            q = deque([i])
            visited[i] = True
            comp = []
            while q:
                v = q.popleft()
                comp.append(v)
                for nb in np.nonzero(mst_cut[v])[0]:
                    if not visited[nb]:
                        visited[nb] = True
                        q.append(nb)
            clusters.append(comp)
    return clusters


def compute_objective(D, clusters):
    obj = 0.0
    assign = np.zeros(D.shape[0], dtype=int)

    for cl in clusters:
        best_j, best_cost = None, float("inf")
        for j in cl:
            cost = D[cl, j].sum()
            if cost < best_cost:
                best_cost = cost
                best_j = j
        obj += best_cost
        for i in cl:
            assign[i] = best_j
    return obj, assign


def plot_clusters(points, assign, k, obj):
    colors = plt.cm.tab10.colors
    plt.figure(figsize=(8, 6))
    for idx, c in enumerate(np.unique(assign)):
        pts = np.where(assign == c)[0]
        col = colors[idx % len(colors)]
        plt.scatter(points[pts, 0], points[pts, 1], color=col, s=60)
        plt.scatter(points[c, 0], points[c, 1],
                    marker="*", s=300, color=col,
                    edgecolor="black")
    plt.title(f"MST heuristic clustering (k={k})\nObjective = {obj:.3f}")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig("heuristic_clusters.png", dpi=300)
    plt.close()


if __name__ == "__main__":
    BASE = Path(__file__).resolve().parent
    points = np.load(BASE / "benchmark_points.npy")

    k = 5
    start = time.time()

    mst, D = build_mst(points)
    mst_cut, _ = cut_longest(mst, k)
    clusters = extract_clusters(mst_cut)
    obj, assign = compute_objective(D, clusters)

    cpu = time.time() - start
    print("MST objective:", obj)
    print("CPU time:", cpu)

    plot_clusters(points, assign, k, obj)
