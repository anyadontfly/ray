import csv
import os
import re
import sys
from typing import Any, Dict, List, Tuple

import matplotlib.pyplot as plt


def parse_filename(filename: str) -> Tuple[int, int, int]:
    """Extract layer_size, num_layers, and rank from filename."""
    pattern = r"_ls(\d+)_nl(\d+)_np(\d+)_rank0\.csv"
    match = re.search(pattern, filename)
    if not match:
        raise ValueError(f"Invalid filename format: {filename}")
    return tuple(map(int, match.groups()))

def read_csv_data(filepath: str) -> Dict[str, int]:
    """Read CSV file and extract name and mean columns."""
    data = {}
    with open(filepath, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            data[row["name"]] = int(float(row["mean"]))
    return data

def process_folder(folder_path: str) -> List[Dict[str, Any]]:
    """Process all CSV files in a folder and return list of Experiment objects."""
    experiments = []

    for filename in os.listdir(folder_path):
        if not filename.endswith("0.csv"):
            continue

        filepath = os.path.join(folder_path, filename)
        layer_size, num_layers, num_partitions = parse_filename(filename)
        data = read_csv_data(filepath)

        exp = {
            "layer_size": layer_size,
            "num_layers": num_layers,
            "num_partitions": num_partitions,
            "data": data,
        }
        experiments.append(exp)

    return experiments

def plot_data(
    experiments_ray: List[Dict[str, Any]],
    experiments_torch: List[Dict[str, Any]],
    layer_size: int,
    num_layers: int
) -> None:
    """Plot the data for each rank."""
    model_size = num_layers * layer_size**2 * 4 / 1024**2

    data_ray = {}
    data_torch = {}

    for exp in experiments_ray:
        num_partitions = exp["num_partitions"]
        total = exp["data"]["total"]
        bucket_size = (num_layers / num_partitions * layer_size**2) * 4 / 1024**2
        data_ray[bucket_size] = total
    
    for exp in experiments_torch:
        num_partitions = exp["num_partitions"]
        total = exp["data"]["total"]
        bucket_size = (num_layers / num_partitions * layer_size**2) * 4 / 1024**2
        data_torch[bucket_size] = total

    # Sort the data by num_partitions
    data_ray = dict(sorted(data_ray.items()))
    data_torch = dict(sorted(data_torch.items()))

    # Plot the data
    plt.figure(figsize=(10, 5))
    plt.plot(data_ray.keys(), data_ray.values(), label="ray", marker='o')
    plt.plot(data_torch.keys(), data_torch.values(), label="torch", marker='o')
    plt.xticks()
    plt.yticks()
    plt.xlabel("Bucket Size (MB)")
    plt.ylabel("Time (ms)")
    plt.title(f"End-2-End Time vs Bucket Size (model_size {model_size} MB)")
    plt.legend()
    plt.grid()
    # save image in current directory
    plt.savefig("utils/e2e time vs bucket size.png")

def main(layer_size: int, num_layers: int, folder_path_ray: str, folder_path_torch: str) -> None:
    experiments_ray = process_folder(folder_path_ray)
    experiments_torch = process_folder(folder_path_torch)
    plot_data(experiments_ray, experiments_torch, layer_size, num_layers)

if __name__ == "__main__":
    layer_size = int(sys.argv[1])
    num_layers = int(sys.argv[2])
    folder_path_ray = "results/barbell/linear/ray/ddp/test_bucket_size/"
    folder_path_torch = "results/barbell/linear/torch/ddp/test_bucket_size/"
    main(layer_size, num_layers, folder_path_ray, folder_path_torch)
