import time

import ray
from ray.dag import InputNode, MultiOutputNode
from ray.experimental.collective import allreduce
from ray.air._internal import torch_utils

import torch


ray.init(num_gpus=2)

torch.manual_seed(42)

@ray.remote
class TensorGenerator:
    def __init__(self, tensor_size):
        self.device = torch_utils.get_devices()[0]
        self.tensor_size = tensor_size

    def generate(self, *args):
        return torch.randn(self.tensor_size).to(self.device)
    
    def generate_ten(self, *args):
        return tuple(torch.randn(self.tensor_size).to(self.device) for _ in range(10))

num_generators = 2
generators = [TensorGenerator.options(num_gpus=1).remote(4096) for _ in range(num_generators)]

with InputNode() as inp:
    all_results = []
    
    for _ in range(10):
        tensors = [generator.generate.bind(inp) for generator in generators]
        all_results.extend(allreduce.bind(tensors))

    dag_no_tuple = MultiOutputNode(all_results)

with InputNode() as inp:
    tensors = [generator.generate_ten.bind(inp) for generator in generators]
    results = allreduce.bind(tensors)
    dag_tuple = MultiOutputNode(results)

compiled_dag = dag_tuple.experimental_compile()
ref = compiled_dag.execute(0)
ray.get(ref)

time_total = 0
for i in range(5):
    time_start = time.perf_counter()
    ref = compiled_dag.execute(0)
    ray.get(ref)
    time_end = time.perf_counter()
    time_total += time_end - time_start
print(f"Avg time: {time_total/5:.4f} s")
