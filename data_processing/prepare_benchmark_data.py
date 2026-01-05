import numpy as np
from pathlib import Path
from scipy.io import arff as arff_io


def pairwise_distances(points):
    diff = points[:, None, :] - points[None, :, :]
    return np.sqrt((diff ** 2).sum(axis=2))


def needs_normalization(X, threshold=10.0):
    """
    Returns True if at least one feature has a scale difference
    greater than `threshold` compared to another feature.
    """
    feature_ranges = X.max(axis=0) - X.min(axis=0)
    non_zero = feature_ranges[feature_ranges > 0]

    if len(non_zero) == 0:
        return False

    return non_zero.max() / non_zero.min() > threshold


def load_arff_without_class(arff_file):
    """
    Load an ARFF file and remove the 'class' attribute if present
    (case-insensitive). Returns a numpy array of features only.
    """
    data, meta = arff_io.loadarff(arff_file)

    attr_names = [name.lower() for name in meta.names()]

    keep_idx = [
        i for i, name in enumerate(attr_names)
        if name != "class"
    ]

    if len(keep_idx) < len(attr_names):
        print("Class attribute detected and removed")

    X = np.array([
        [float(row[i]) for i in keep_idx]
        for row in data
    ])

    return X


def write_arff(filename, X, relation_name="normalized_data"):
    """
    Write a numpy array X to an ARFF file.
    """
    filename = Path(filename)
    filename.parent.mkdir(parents=True, exist_ok=True)

    with open(filename, "w") as f:
        f.write(f"@RELATION {relation_name}\n\n")

        d = X.shape[1]
        for j in range(d):
            f.write(f"@ATTRIBUTE a{j} REAL\n")

        f.write("\n@DATA\n")
        for row in X:
            f.write(",".join(f"{v:.6f}" for v in row) + "\n")


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
    
    arff_file = Path(arff_file)

    print(f"Using dataset: {arff_file}")

    # Load ARFF data and remove class label if present
    X_raw = load_arff_without_class(arff_file)

    print("Original data dimension:", X_raw.shape[1])
    print("Original feature ranges:",
          X_raw.max(axis=0) - X_raw.min(axis=0))

    # Automatic normalization if scale mismatch detected
    if needs_normalization(X_raw):
        print("Normalizing features (scale mismatch detected)")
        mean = X_raw.mean(axis=0)
        std = X_raw.std(axis=0)
        std[std == 0] = 1.0  # safety for constant features
        X = (X_raw - mean) / std
        normalized = True
    else:
        print("No normalization needed")
        X = X_raw
        normalized = False

    print("Post-processing feature means:", X.mean(axis=0))
    print("Post-processing feature stds :", X.std(axis=0))

    # Save ARFF for inspection
    suffix = "_normalized" if normalized else "_raw"
    write_arff(
        base_dir / "normalized_datasets" / (arff_file.stem + suffix + ".arff"),
        X,
        relation_name=arff_file.stem + suffix
    )

    # Subsampling
    rng = np.random.default_rng(seed)
    m = int(round(sample_ratio * len(X)))
    idx = np.sort(rng.choice(len(X), size=m, replace=False))

    points = X[idx]
    D = pairwise_distances(points)

    # Write AMPL .dat file
    lines = []
    lines.append("# Benchmark data for cluster-median\n")
    lines.append(f"# Sample ratio: {sample_ratio}, seed={seed}\n")
    lines.append(f"param m := {m};\n")
    lines.append(f"param k := {k};\n\n")
    lines.append("param d :\n")
    lines.append("    " + " ".join(str(j + 1) for j in range(m)) + " :=\n")

    for i in range(m):
        lines.append(
            f"{i + 1} " + " ".join(f"{D[i, j]:.6f}" for j in range(m)) + "\n"
        )

    lines.append(";\n")

    (base_dir / "benchmark.dat").write_text("".join(lines))

    return points, D, m
