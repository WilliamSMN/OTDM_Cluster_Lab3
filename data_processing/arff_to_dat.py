import numpy as np
from scipy.io import arff
import os

def euclidean_distance(point1, point2):
    """Calculate Euclidean distance between two points."""
    return np.sqrt(np.sum((point1 - point2) ** 2))

def compute_distance_matrix(data):
    """
    Compute the distance matrix for all points.
    
    Args:
        data: numpy array of shape (m, n_features)
    
    Returns:
        distance_matrix: numpy array of shape (m, m)
    """
    m = len(data)
    distance_matrix = np.zeros((m, m))
    
    for i in range(m):
        for j in range(m):
            distance_matrix[i, j] = euclidean_distance(data[i], data[j])
    
    return distance_matrix

def load_arff_dataset(arff_file):
    """
    Load an ARFF file and extract numeric features.
    
    Args:
        arff_file: path to the ARFF file
    
    Returns:
        data: numpy array of numeric features
        true_labels: array of class labels (if available)
        n_classes: number of unique classes (for k value)
    """
    data_arff, meta = arff.loadarff(arff_file)
    
    # Convert to numpy array
    data_list = []
    labels = []
    
    for row in data_arff:
        # Extract numeric attributes (exclude CLASS if present)
        numeric_values = []
        for i, name in enumerate(meta.names()):
            if name.upper() not in ['CLASS', 'CLUSTER']:
                numeric_values.append(float(row[i]))
            else:
                labels.append(row[i])
        
        data_list.append(numeric_values)
    
    data = np.array(data_list)
    
    # Determine number of classes (for k)
    if labels:
        # Handle both numeric and byte string labels
        unique_labels = set()
        for label in labels:
            if isinstance(label, bytes):
                unique_labels.add(label.decode('utf-8'))
            else:
                unique_labels.add(str(label))
        n_classes = len(unique_labels)
    else:
        n_classes = None
    
    return data, labels, n_classes

def sample_dataset(data, labels, sample_proportion=1.0, random_seed=None):
    """
    Sample a proportion of the dataset.
    
    Args:
        data: numpy array of features
        labels: list of labels
        sample_proportion: float between 0 and 1 (proportion to keep)
        random_seed: int for reproducibility (optional)
    
    Returns:
        sampled_data: sampled data array
        sampled_labels: sampled labels list
    """
    if sample_proportion <= 0 or sample_proportion > 1:
        raise ValueError("sample_proportion must be between 0 and 1")
    
    if sample_proportion == 1.0:
        return data, labels
    
    # Set random seed for reproducibility
    if random_seed is not None:
        np.random.seed(random_seed)
    
    m = len(data)
    sample_size = int(m * sample_proportion)
    
    # Random sampling without replacement
    indices = np.random.choice(m, size=sample_size, replace=False)
    indices = np.sort(indices)  # Sort to maintain some order
    
    sampled_data = data[indices]
    sampled_labels = [labels[i] for i in indices] if labels else []
    
    return sampled_data, sampled_labels

def generate_ampl_data(arff_file, output_file='cluster_median.dat', k=None, 
                       sample_proportion=1.0, random_seed=None):
    """
    Generate the AMPL data file from an ARFF dataset.
    
    Args:
        arff_file: path to input ARFF file
        output_file: path to output .dat file
        k: number of clusters (if None, use number of classes from dataset)
        sample_proportion: proportion of dataset to use (0 < x <= 1)
        random_seed: random seed for sampling reproducibility
    """
    # Load dataset
    data, labels, n_classes = load_arff_dataset(arff_file)
    
    print(f"Original dataset: {len(data)} points")
    
    # Sample dataset if needed
    if sample_proportion < 1.0:
        data, labels = sample_dataset(data, labels, sample_proportion, random_seed)
        print(f"Sampled dataset: {len(data)} points ({sample_proportion*100:.1f}%)")
    
    # Use k from dataset if not specified
    if k is None:
        if n_classes is not None:
            k = n_classes
            print(f"Using k={k} clusters (detected from dataset)")
        else:
            raise ValueError("Cannot determine k. Please specify k parameter.")
    
    # Compute distance matrix
    print(f"Computing distance matrix for {len(data)} points...")
    distance_matrix = compute_distance_matrix(data)
    
    m = len(data)
    
    # Generate AMPL data file
    with open(output_file, 'w') as f:
        f.write(f"# Data file for cluster-median problem\n")
        f.write(f"# Generated from: {arff_file}\n")
        if sample_proportion < 1.0:
            f.write(f"# Sampled: {sample_proportion*100:.1f}% of original data")
            if random_seed is not None:
                f.write(f" (seed={random_seed})")
            f.write("\n")
        f.write("\n")
        
        f.write(f"param m := {m};\n")
        f.write(f"param k := {k};\n\n")
        
        f.write("param d:\n")
        # Write column headers
        f.write("     ")
        for j in range(1, m + 1):
            f.write(f"{j:12}")
        f.write(" :=\n")
        
        # Write distance matrix
        for i in range(m):
            f.write(f"{i+1:3} ")
            for j in range(m):
                f.write(f"{distance_matrix[i, j]:12.6f}")
            f.write("\n")
        
        f.write(";\n")
    
    print(f"AMPL data file created: {output_file}")
    print(f"Dataset info: m={m} points, k={k} clusters")
    print(f"Feature dimensions: {data.shape[1]}")

def process_arff_to_ampl(arff_file, k=None, output_prefix=None, 
                         sample_proportion=1.0, random_seed=None):
    """
    Complete pipeline: convert ARFF to AMPL model and data files.
    
    Args:
        arff_file: path to input ARFF file
        k: number of clusters (if None, use from dataset)
        output_prefix: prefix for output files (default: based on arff filename)
        sample_proportion: proportion of dataset to use (0 < x <= 1)
        random_seed: random seed for sampling reproducibility
    """
    if output_prefix is None:
        base_name = os.path.splitext(os.path.basename(arff_file))[0]
        output_prefix = base_name
    
    dat_file = f"{output_prefix}.dat"
    
    generate_ampl_data(arff_file, dat_file, k, sample_proportion, random_seed)
    
    print(f"\n✓ Conversion complete!")
    print(f"  Data file: {dat_file}")

if __name__ == "__main__":
    # Here the ARFF files are downloaded from https://github.com/deric/clustering-benchmark/tree/master/src/main/resources/datasets
    arff_file = "./datasets/2d-3c-no123.arff"
    
    # Examples of usage:
    
    # 1. Use full dataset (default)
    # process_arff_to_ampl(arff_file)
    
    # 2. Use 50% of the dataset
    # process_arff_to_ampl(arff_file, sample_proportion=0.5)
    
    # 3. Use 25% of the dataset with reproducible results
    process_arff_to_ampl(arff_file, sample_proportion=0.10, random_seed=42)
