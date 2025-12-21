from amplpy import AMPL
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import os


def cluster_median_plot(cluster_assignments, cluster_medians, k, obj_value):
    """
    Create visualization plots for clustering results.
    
    Args:
        cluster_assignments: array of cluster assignments for each point
        cluster_medians: list of median point indices
        k: number of clusters
        obj_value: objective function value
    """
    # Prepare output folder
    output_dir = "./clustering_results"
    os.makedirs(output_dir, exist_ok=True)
    
    unique_clusters, counts = np.unique(cluster_assignments, return_counts=True)
    
    # ---------- 1) Cluster Size Distribution ----------
    plt.figure(figsize=(10, 6))
    plt.bar(unique_clusters, counts, color='steelblue', alpha=0.7, edgecolor='black')
    plt.xlabel('Cluster ID (Median Point Index)', fontsize=12)
    plt.ylabel('Number of Points', fontsize=12)
    plt.title(f'Cluster Size Distribution (k={k}, Total Distance={obj_value:.2f})', fontsize=14)
    plt.grid(axis='y', alpha=0.3)
    plt.savefig(f"{output_dir}/01_cluster_sizes.png", dpi=300, bbox_inches='tight')
    plt.show()


def cluster_median_analysis(ampl):
    """
    Analyze clustering results from AMPL solution.
    
    Args:
        ampl: AMPL instance with solved model
    """
    # Get parameters
    m = int(ampl.getParameter("m").value())
    k = int(ampl.getParameter("k").value())
    
    # Retrieve solution: x[i,j]
    x_data = ampl.get_data("x").toDict()
    
    # Build cluster assignments
    cluster_assignments = np.zeros(m, dtype=int)
    cluster_medians = []
    
    for (i, j), val in x_data.items():
        if val > 0.5:  # x[i,j] = 1
            cluster_assignments[i-1] = j  # i-th point belongs to cluster-j
            if i == j:  # This is a median
                cluster_medians.append(j)
    
    # Get objective value
    obj_value = ampl.get_value("TotalDistance")
    
    print("\n=== Cluster-Median Results ===")
    print(f"Total Distance (Objective): {obj_value:.4f}")
    print(f"Number of clusters formed: {len(cluster_medians)}")
    print(f"Cluster medians (point indices): {sorted(cluster_medians)}")
    
    # Count points per cluster
    unique_clusters, counts = np.unique(cluster_assignments, return_counts=True)
    print("\nCluster sizes:")
    for cluster_id, count in zip(unique_clusters, counts):
        print(f"  Cluster {cluster_id}: {count} points (median: point {cluster_id})")
    
    cluster_median_plot(cluster_assignments, cluster_medians, k, obj_value)


def run_cluster_median():
    ampl = AMPL()

    ampl.read("cluster_median.mod")
    ampl.readData("2d-3c-no123.dat")
    ampl.setOption("solver", "cplex")

    print("--- Solving ---")
    ampl.solve()

    print("\n--- Generating Diagnostic Plots ---")
    cluster_median_analysis(ampl)

    ampl.close()


if __name__ == "__main__":
    run_cluster_median()