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
    
    def generate_n(self, n):
        return tuple(torch.randn(self.tensor_size).to(self.device) for _ in range(n))

num_generators = 2
num_tensors = [10, 15, 20, 25, 30]
generators = [TensorGenerator.options(num_gpus=1).remote((4096, 8)) for _ in range(num_generators)]


USE_TUPLE = False

if USE_TUPLE:
    with InputNode() as inp:
        tensors = [generator.generate_n.bind(inp) for generator in generators]
        results = allreduce.bind(tensors)
        dag_tuple = MultiOutputNode(results)
    
    compiled_dag = dag_tuple.experimental_compile()
    times = {}
    for num_tensor in num_tensors:
        # warm up
        ref = compiled_dag.execute(num_tensor)
        ray.get(ref)

        time_total = 0
        for i in range(5):
            time_start = time.perf_counter()
            ref = compiled_dag.execute(num_tensor)
            ray.get(ref)
            time_end = time.perf_counter()
            time_total += time_end - time_start
        times[num_tensor] = round(time_total / 5, 4)
    for key, value in times.items():
        print(f"Num generations: {key}, Avg time: {value} s")
else:
    times = {}
    for num_tensor in num_tensors:

        with InputNode() as inp:
            all_results = []
            
            for _ in range(num_tensor):
                tensors = [generator.generate.bind(inp) for generator in generators]
                all_results.extend(allreduce.bind(tensors))

            dag_no_tuple = MultiOutputNode(all_results)
        
        compiled_dag = dag_no_tuple.experimental_compile()
        ref = compiled_dag.execute(None)
        ray.get(ref)

        time_total = 0
        for i in range(5):
            time_start = time.perf_counter()
            ref = compiled_dag.execute(None)
            ray.get(ref)
            time_end = time.perf_counter()
            time_total += time_end - time_start
        times[num_tensor] = round(time_total / 5, 4)
        compiled_dag.teardown()
    for key, value in times.items():
        print(f"Num generations: {key}, Avg time: {value} s")
