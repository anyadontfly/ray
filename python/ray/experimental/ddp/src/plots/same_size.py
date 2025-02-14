import os
from typing import Any, Dict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import font_manager

pwd = "/home/wxdeng/Ray-Sea2Sky/python/ray/experimental/ddp"
version = "0111_s250_e50"
# version = "0111_s500_e50"

args: Dict[str, Any] = {}
args["layer_size_values"] = [2560, 1280, 640, 512, 320, 160, 80]
args["num_layers_values"] = [10, 40, 160, 250, 640, 2560, 10240]
# args["num_layers_values"] = [20, 80, 320, 500, 1280, 5120, 20480]
args["torch_ddp"] = f"{pwd}/results/xuhui/torch_ddp/grids/{version}"
args["ray_no_allreduce"] = f"{pwd}/results/xuhui/ray_no_allreduce/grids/{version}"
args["ray_bucketing"] = f"{pwd}/results/xuhui/ray_bucketing_unify/grids/{version}"
args["ray_bucketing_overlapping"] = (
    f"{pwd}/results/xuhui/ray_bucketing_overlapping_unify/grids/{version}"
)
args["table"] = f"{pwd}/results/xuhui/same_size/{version}/ddp.csv"
args["png"] = f"{pwd}/results/xuhui/same_size/{version}/ddp.png"
args["pdf"] = f"{pwd}/results/xuhui/same_size/{version}/ddp.pdf"

keys = [
    "torch_ddp",
    "ray_no_allreduce",
    "ray_bucketing",
    "ray_bucketing_overlapping",
]
for key in keys:
    assert os.path.exists(args[key]), args[key]
os.makedirs(os.path.dirname(args["table"]), exist_ok=True)

data = []
for ls, nl in zip(args["layer_size_values"], args["num_layers_values"]):
    if ls < 320:
        continue
    row = {
        "layer_size": ls,
        "num_layers": nl,
    }
    filename = f"ls{ls}_nl{nl}_rank0.csv"

    for key in keys:
        filepath = os.path.join(args[key], filename)
        assert os.path.exists(filepath), filepath
        df = pd.read_csv(filepath)
        total = df.iloc[0]["mean"]
        row[key] = total

    for i in range(1, 4):
        row[f"relative_{keys[i]}"] = round(row[keys[i]] / row[keys[0]], 2)
    row[f"speedup_overlap"] = round(row[keys[2]] / row[keys[3]], 2)
    data.append(row)

df = pd.DataFrame(data)

output_columns = [
    "layer_size",
    "num_layers",
    "torch_ddp",
    "relative_ray_no_allreduce",
    "relative_ray_bucketing",
    "relative_ray_bucketing_overlapping",
    "speedup_overlap",
]
df[output_columns].to_csv(args["table"], index=False)

font_paths = ["/usr/share/fonts/truetype/roboto"]
font_files = font_manager.findSystemFonts(fontpaths=font_paths)
for font_file in font_files:
    font_manager.fontManager.addfont(font_file)
plt.rcParams["font.family"] = "Roboto"
fontsize_large = 17
fontsize_medium = 13
fontsize_small = 11
fontsize_tiny = 9
plt.rcParams.update(
    {
        "font.size": fontsize_large,
        "axes.labelsize": fontsize_large,
        "axes.titlesize": fontsize_large,
        "xtick.labelsize": fontsize_medium,
        "ytick.labelsize": fontsize_medium,
        "legend.fontsize": fontsize_large,
    }
)

fig, ax1 = plt.subplots(figsize=(10, 6))

ax2 = ax1.twinx()

x = np.arange(len(df))
width = 0.23

bars1 = ax1.bar(
    x - width,
    df["relative_ray_bucketing"],
    width,
    label="Ray CD w/ Bucketing",
    color="#11616B",
    edgecolor="black",
    linewidth=1,
)

bars2 = ax1.bar(
    x,
    df["relative_ray_bucketing_overlapping"],
    width,
    label="Ray CD w/ Bucketing and Overlap",
    color="#7BBDB6",
    edgecolor="black",
    linewidth=1,
)

bars3 = ax2.bar(
    x + width,
    df["speedup_overlap"],
    width,
    label="Overlap Speedup",
    color="#C2CFAF",
    edgecolor="black",
    linewidth=1,
)

ax1.set_ylabel("Relative Latency to Torch DDP")
ax1.set_ylim(0, 3.5)

ax2.set_ylabel("Overlap Speedup")
ax2.tick_params(axis="y")
ax2.set_ylim(1.0, 2.0)

plt.xticks(
    x,
    [
        f'({int(row["layer_size"])}, {int(row["num_layers"])})'
        for _, row in df.iterrows()
    ],
    rotation=45,
)
ax1.set_xlabel("(Hidden Dimension, Number of Layers)")

lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right")

plt.title("Microbenchmark of DDP")

plt.tight_layout()

plt.savefig(args["png"], bbox_inches="tight")
plt.savefig(args["pdf"], bbox_inches="tight")
