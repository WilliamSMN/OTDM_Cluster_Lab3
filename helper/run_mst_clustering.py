import numpy as np
import matplotlib.pyplot as plt
from scipy.io import arff
from scipy.spatial.distance import pdist, squareform
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import minimum_spanning_tree
from collections import deque
from pathlib import Path


# ============================================================
# DATA LOADING (ARFF)
# ============================================================

def load_arff_first_m_points(arff_file, m):
    """
    Load first m points (a0, a1) from ARFF file, ignore 'class'.
    """
    data_arff, meta = arff.loadarff(arff_file)

    points = []
    for i, row in enumerate(data_arff):
        if i >= m:
            break
        points.append([float(row[0]), float(row[1])])  # a0, a1 only

    return np.array(points)


# ============================================================
# MST + CUT
# ============================================================

def build_mst(points):
    D = squareform(pdist(points, metric="euclidean"))
    mst_dir = minimum_spanning_tree(csr_matrix(D)).toarray()
    mst = mst_dir + mst_dir.T
    return mst, D


def cut_k_minus_1_longest_edges(mst, k):
    n = mst.shape[0]
    mst_cut = mst.copy()

    edges = []
    for i in range(n):
        for j in range(i + 1, n):
            if mst[i, j] > 0:
                edges.append((i, j, mst[i, j]))

    edges.sort(key=lambda x: x[2], reverse=True)
    removed_edges = edges[:k - 1]

    for i, j, _ in removed_edges:
        mst_cut[i, j] = 0
        mst_cut[j, i] = 0

    return mst_cut, removed_edges


# ============================================================
# CLUSTERS
# ============================================================

def extract_clusters(mst_cut):
    n = mst_cut.shape[0]
    visited = [False] * n
    clusters = []

    for start in range(n):
        if not visited[start]:
            queue = deque([start])
            visited[start] = True
            component = []

            while queue:
                node = queue.popleft()
                component.append(node)
                neighbors = np.nonzero(mst_cut[node])[0]
                for nb in neighbors:
                    if not visited[nb]:
                        visited[nb] = True
                        queue.append(nb)

            clusters.append(component)

    return clusters


# ============================================================
# TRUE CLUSTER MEDIAN + OBJECTIVE
# ============================================================

def compute_cluster_medians_and_objective(points, clusters, D):
    total_objective = 0.0
    medians = []

    for cluster in clusters:
        best_point = None
        best_cost = float("inf")

        for j in cluster:
            cost = sum(D[i, j] for i in cluster)
            if cost < best_cost:
                best_cost = cost
                best_point = j

        medians.append(best_point)
        total_objective += best_cost

    return medians, total_objective


# ============================================================
# VISUALIZATION
# ============================================================

def plot_before_after(points, mst, mst_cut, removed_edges,
                      clusters, medians, k, objective_value):

    colors = plt.cm.tab10.colors
    n = points.shape[0]

    plt.figure(figsize=(16, 6))

    # ---------- BEFORE ----------
    plt.subplot(1, 2, 1)
    for i in range(n):
        for j in range(i + 1, n):
            if mst[i, j] > 0:
                plt.plot(
                    [points[i, 0], points[j, 0]],
                    [points[i, 1], points[j, 1]],
                    color="gray",
                    alpha=0.7
                )

    for i, j, _ in removed_edges:
        plt.plot(
            [points[i, 0], points[j, 0]],
            [points[i, 1], points[j, 1]],
            color="red",
            linewidth=3
        )

    plt.scatter(points[:, 0], points[:, 1], s=60, color="black")
    plt.title(f"MST BEFORE cut (k={k})\nRed = removed edges")
    plt.grid(alpha=0.3)

    # ---------- AFTER ----------
    plt.subplot(1, 2, 2)
    for i in range(n):
        for j in range(i + 1, n):
            if mst_cut[i, j] > 0:
                plt.plot(
                    [points[i, 0], points[j, 0]],
                    [points[i, 1], points[j, 1]],
                    color="gray",
                    alpha=0.7
                )

    for idx, cluster in enumerate(clusters):
        col = colors[idx % len(colors)]
        pts = points[cluster]

        plt.scatter(
            pts[:, 0], pts[:, 1],
            color=col, s=70,
            label=f"Cluster {idx+1}"
        )

        m = medians[idx]
        plt.scatter(
            points[m, 0], points[m, 1],
            marker="*",
            s=300,
            color=col,
            edgecolor="black",
            linewidth=1.5,
            zorder=5
        )

    plt.title(
        f"MST AFTER cut → {len(clusters)} clusters\n"
        f"Cluster-median objective = {objective_value:.3f}"
    )
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    BASE_DIR = Path(__file__).resolve().parent

    arff_file = BASE_DIR / "data_processing" / "datasets" / "2d-3c-no123.arff"

    m = 725   # number of points (MANUAL)
    k = 3       # number of clusters

    points = load_arff_first_m_points(arff_file, m)

    mst, D = build_mst(points)
    mst_cut, removed_edges = cut_k_minus_1_longest_edges(mst, k)

    clusters = extract_clusters(mst_cut)

    medians, objective_value = compute_cluster_medians_and_objective(
        points, clusters, D
    )

    print(f"Number of points m = {m}")
    print(f"Number of clusters k = {k}")
    print(f"Cluster-median objective value = {objective_value:.6f}")

    for i, c in enumerate(clusters, 1):
        print(f"Cluster {i}: size={len(c)}, median point index={medians[i-1]}")

    plot_before_after(
        points,
        mst,
        mst_cut,
        removed_edges,
        clusters,
        medians,
        k,
        objective_value
    )
