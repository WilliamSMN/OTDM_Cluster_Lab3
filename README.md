# OTDM – Laboratory 3  
## The Cluster-Median Problem

This project implements and compares two approaches for solving the **cluster-median problem**:

- an **exact integer programming formulation** solved with **AMPL + CPLEX**
- a **heuristic method based on a Minimum Spanning Tree (MST)**

Both approaches are applied to artificial and real-world datasets provided in ARFF format.

## How to Run

Make sure Python and AMPL are correctly installed and accessible.

From the project root directory, run:

```bash
python benchmark_cluster_median.py
