# Cluster-Median Problem
# Parameters
param m > 0 integer;  # number of elements/points
param k > 0 integer;  # number of clusters desired
param d{1..m, 1..m} >= 0;  # distance matrix

# Decision variables
var x{1..m, 1..m} binary; 

# Objective: minimize total distance from points to their cluster medians
minimize TotalDistance:
    sum{i in 1..m, j in 1..m} d[i,j] * x[i,j];

# Every point belongs to exactly one cluster
subject to OneCluster{i in 1..m}:
    sum{j in 1..m} x[i,j] = 1;

# Exactly k clusters must be created
subject to ExactlyKClusters:
    sum{j in 1..m} x[j,j] = k;

# A point can belong to cluster-j only if cluster-j exists (i.e., x[j,j] = 1)
subject to ClusterExists{j in 1..m}:
    m * x[j,j] >= sum{i in 1..m} x[i,j];