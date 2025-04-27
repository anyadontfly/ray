import torch

import ray
from ray.dag import InputNode, MultiOutputNode
from ray.experimental.collective import allreduce


@ray.remote(num_gpus=1)
class Actor:
    def __init__(self, pool_size):
        torch.cuda.profiler.start()
        self.deviece = "cuda:0"
        self.comm_tensors = [
            torch.randn(1000).to(self.deviece) for _ in range(pool_size)
        ]
        self.offset = 0

    def start_trace(self, _):
        return 1

    def send_reduce(self, num_tensors):
        tensors = self.comm_tensors[self.offset : self.offset + num_tensors]
        self.offset += num_tensors
        if self.offset >= len(self.comm_tensors):
            self.offset = 0
        return tuple(tensors)
    
    def end_trace(self, *args):
        torch.cuda.profiler.stop()
        return 1
    

bucket_sizes = [100]  # in number of tensors
num_tensors = 10_000
actors = [Actor.options(num_gpus=1).remote(num_tensors) for _ in range(2)]

for bucket_size in bucket_sizes:

    with InputNode() as inp:
        res = []
        start_traces = [actor.start_trace.bind(inp) for actor in actors]
        for _ in range(num_tensors // bucket_size):
            sends = [actor.send_reduce.bind(inp) for actor in actors]
            res += allreduce.bind(sends)
        end_traces = [actor.end_trace.bind(start) for actor, start in zip(actors, start_traces)]
        dag = MultiOutputNode(end_traces+res)

    compiled_dag = dag.experimental_compile(_overlap_gpu_communication=True)

    iter = 10
    for _ in range(iter):
        times = ray.get(compiled_dag.execute(bucket_size))
