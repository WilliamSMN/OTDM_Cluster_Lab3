import numpy as np
from pathlib import Path
from scipy.io import arff as arff_io


def pairwise_distances(points):
    diff = points[:, None, :] - points[None, :, :]
    return np.sqrt((diff ** 2).sum(axis=2))


def generate_benchmark_data(
    sample_ratio: float,
    k: int,
    seed: int = 42,
    arff_file: Path | None = None,
    base_dir: Path | None = None
):
    if base_dir is None:
        base_dir = Path(__file__).resolve().parent

    if arff_file is None:
        arff_file = base_dir / "data_processing/datasets/2d-3c-no123.arff"

    # Load data
    data, _ = arff_io.loadarff(arff_file)
    points_all = np.array([[float(r[0]), float(r[1])] for r in data])

    rng = np.random.default_rng(seed)
    m = int(round(sample_ratio * len(points_all)))
    idx = np.sort(rng.choice(len(points_all), size=m, replace=False))

    points = points_all[idx]
    D = pairwise_distances(points)

    # Save points
    #np.save(base_dir / "benchmark_points.npy", points)

    # Write AMPL .dat
    lines = []
    lines.append("# Benchmark data for cluster-median\n")
    lines.append(f"# Sample ratio: {sample_ratio}, seed={seed}\n")
    lines.append(f"param m := {m};\n")
    lines.append(f"param k := {k};\n\n")
    lines.append("param d :\n")
    lines.append("    " + " ".join(str(j+1) for j in range(m)) + " :=\n")
    for i in range(m):
        lines.append(
            f"{i+1} " + " ".join(f"{D[i,j]:.6f}" for j in range(m)) + "\n"
        )
    lines.append(";\n")

    (base_dir / "benchmark.dat").write_text("".join(lines))

    return points, D, m
