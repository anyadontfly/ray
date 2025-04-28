import torch

import ray
from ray.dag import InputNode, MultiOutputNode
from ray.experimental.collective import allreduce


@ray.remote(num_gpus=1)
class Actor:
    def __init__(self):
        self.device = "cuda:0"

    def start_trace(self, _):
        torch.cuda.profiler.start()
        return 1

    def send_reduce(self, num_tensors):
        tensors = []
        for _ in range(num_tensors):
            tensors.append(torch.ones(1000, device=self.device))
        return tuple(tensors)
    
    def end_trace(self, *args):
        torch.cuda.profiler.stop()
        return 1
    

bucket_size = 10  # in number of tensors, number of params is bucket_size * 1000
num_tensors = 10_000  # number of tensors need to time 1000
actors = [Actor.options(num_gpus=1).remote() for _ in range(2)]


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
