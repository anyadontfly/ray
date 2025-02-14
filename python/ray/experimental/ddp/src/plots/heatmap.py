import os
from typing import Any, Dict

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

args: Dict[str, Any] = {}
args["layer_size_values"] = [40, 80, 160, 320, 640, 1280, 2560]
args["num_layers_values"] = [4, 8, 16, 32, 64, 128, 256]
args["path_torch_ddp"] = "results/xuhui/torch_ddp/grids/0109"
args["path_ray_bucketing"] = "results/xuhui/ray_bucketing/grids/0109"
args["path_heatmap"] = "results/xuhui/heatmap/0109/ddp.png"

# Read CSV files
data = []
for ls in args["layer_size_values"]:
    for nl in args["num_layers_values"]:
        row = {
            "layer_size": ls,
            "num_layers": nl,
        }
        filename = f"ls{ls}_nl{nl}_rank0.csv"

        filepath = os.path.join(args["path_torch_ddp"], filename)
        assert os.path.exists(filepath), filepath
        df = pd.read_csv(filepath)
        torch_ddp = df.iloc[0]["mean"]

        filepath = os.path.join(args["path_ray_bucketing"], filename)
        assert os.path.exists(filepath), filepath
        df = pd.read_csv(filepath)
        ray_bucketing = df.iloc[0]["mean"]

        row["relative_ray_bucketing"] = ray_bucketing / torch_ddp

        data.append(row)

# Convert to DataFrame
df = pd.DataFrame(data)

# Create heatmaps
plt.figure(figsize=(12, 5))

# Table 1: relative_ray_bucketing
plt.subplot(1, 2, 1)
pivot_ray_bucketing = df.pivot(
    index="layer_size",
    columns="num_layers",
    values="relative_ray_bucketing",
).sort_index(ascending=False)
sns.heatmap(
    pivot_ray_bucketing,
    annot=True,
    fmt=".2f",
    cmap="RdBu_r",
    center=0,
)
plt.title("Relative Time (Ray with Bucketing / Torch DDP)")
plt.yticks(rotation=0)

# Table 2: relative_ray_bucketing_overlapping
plt.subplot(1, 2, 2)
pivot_ray_bucketing = df.pivot(
    index="layer_size",
    columns="num_layers",
    values="relative_ray_bucketing",
).sort_index(ascending=False)
# sns.heatmap(
#     pivot_ray_bucketing,
#     annot=True,
#     fmt=".2f",
#     cmap="RdBu_r",
#     center=0,
# )
plt.title("Relative Time (Ray with Bucketing and Overlapping / Torch DDP)")
plt.yticks(rotation=0)

plt.tight_layout()
os.makedirs(os.path.dirname(args["path_heatmap"]), exist_ok=True)
plt.savefig(args["path_heatmap"])
