import csv
import os
import re
import argparse
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

def plot_data(experiments: List[Dict[str, Any]], layer_size: int, num_layers: int, folder_path: str) -> None:
    """Plot the data for each rank."""
    data = {}

    for exp in experiments:
        num_partitions = exp["num_partitions"]
        total = exp["data"]["total"]

        bucket_size = num_layers / num_partitions * layer_size**2
        data[bucket_size] = total

    # Sort the data by num_partitions
    data = dict(sorted(data.items()))

    # Plot the data
    plt.figure(figsize=(10, 5))
    plt.plot(data.keys(), data.values(), label="Rank 0", marker='o')
    plt.xlabel("Number of elements")
    plt.ylabel("Time (micro sconds)")
    plt.title(f"End-2-End Time vs Bucket Size (layer_size={layer_size}, num_layers={num_layers})")
    plt.legend()
    plt.grid()
    # save the data in given folder
    plt.savefig(os.path.join(folder_path, "bandwidth_vs_partitions.png"))

def main(layer_size: int, num_layers: int, folder_path: str) -> None:
    experiments = process_folder(folder_path)
    plot_data(experiments, layer_size, num_layers, folder_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--folder-path",
        type=str,
    )
    parser.add_argument(
        "--layer-size",
        type=int,
    )
    parser.add_argument(
        "--num-layers",
        type=int,
    )

    args = parser.parse_args()
    folder_path = args.folder_path
    layer_size = args.layer_size
    num_layers = args.num_layers

    if not os.path.exists(folder_path):
        raise ValueError(f"Folder {folder_path} does not exist.")
    if not os.path.isdir(folder_path):
        raise ValueError(f"{folder_path} is not a directory.")
    
    main(layer_size, num_layers, folder_path)
