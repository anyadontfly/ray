from collections import defaultdict
import time

import ray
from ray.dag import InputNode, MultiOutputNode
from ray.experimental.collective import allreduce
from ray.air._internal import torch_utils

import torch

import matplotlib.pyplot as plt


ray.init(num_gpus=2)

torch.manual_seed(42)

@ray.remote
class TensorGenerator:
    def __init__(self, tensor_size, n_tensors):
        self.device = torch_utils.get_devices()[0]
        self.tensor_size = tensor_size
        # initialize a pool of tensors so that time is not spent on initialization during dag execution
        self.tensors = set(torch.randn(self.tensor_size).to(self.device) for _ in range(n_tensors))

    def get_tensor(self, *args):
        # pop a tensor from the pool
        return self.tensors.pop()
    
    def get_tuple(self, n):
        # pop n tensors from the pool
        return tuple(self.tensors.pop() for _ in range(n))
    
    # recv and refill tensors that have been popped
    def recv(self, inp):
        lst = [inp] if isinstance(inp, torch.Tensor) else list(inp)
        for tensor in lst:
            self.tensors.add(tensor)
        return len(lst)

# number of actors
num_generators = 2
# number of tensors to allreduce
num_tensors_lst = [32, 64, 128]
# bucket sizes
bucket_sizes = [1, 2, 4, 8, 16, 32]
# shape of each tensor
tensor_shape = (4096, 32)

# two actors, one for each GPU
generators = [TensorGenerator.options(num_gpus=1).remote(tensor_shape, 50) for _ in range(num_generators)]

times = defaultdict(list)


for num_tensors in num_tensors_lst:
    for bucket_size in bucket_sizes:
        # number of allreduce operations
        num_ar = int(num_tensors / bucket_size)

        with InputNode() as inp:
            inp0, inp1 = inp, inp
            for _ in range(num_ar):
                # if bucket_size is 1, use get_tensor, else use get_tuple
                if bucket_size == 1:
                    tensors = [
                        generator.get_tensor.bind(_inp)
                        for generator, _inp in zip(generators, [inp0, inp1])
                    ]
                else:
                    tensors = [
                        generator.get_tuple.bind(_inp)
                        for generator, _inp in zip(generators, [inp0, inp1])
                    ]
                results = allreduce.bind(tensors)
                inp0, inp1 = [
                    generator.recv.bind(result)
                    for generator, result in zip(generators, results)
                ]

            dag = MultiOutputNode([inp0, inp1])

        # first time of running compiled dag is not counted
        compiled_dag = dag.experimental_compile()
        ref = compiled_dag.execute(bucket_size)
        ray.get(ref)

        # average time of 5 runs
        time_total = 0
        for i in range(5):
            time_start = time.perf_counter()
            ref = compiled_dag.execute(bucket_size)
            ray.get(ref)
            time_end = time.perf_counter()
            time_total += (time_end - time_start)
        times[num_tensors].append(time_total / 5 * 1000)
        
        compiled_dag.teardown()

for num_tensors in times.keys():
    plt.plot(bucket_sizes, times[num_tensors], label=f"{num_tensors} tensors")
plt.legend()
plt.xlabel("Bucket size")
plt.ylabel("Time (ms)")
plt.title(f"tensor_size={tensor_shape}")
plt.savefig("python/ray/dag/tests/experimental/allreduce_tuple_benchmark/allreduce_tuple_benchmark.png")
