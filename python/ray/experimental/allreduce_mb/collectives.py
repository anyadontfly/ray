import time

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
            torch.randn(100).to(self.deviece) for _ in range(pool_size)
        ]
        self.offset = 0

    def start_trace(self, _):
        torch.cuda.synchronize()
        self.event_start = torch.cuda.Event(enable_timing=True)
        self.event_start.record()
        return 1

    def send_reduce(self, size):
        tensors = self.comm_tensors[self.offset : self.offset + size]
        self.offset += size
        if self.offset >= len(self.comm_tensors):
            self.offset = 0
        return tuple(tensors)
    
    def end_trace(self, *args):
        torch.cuda.synchronize()
        self.event_end = torch.cuda.Event(enable_timing=True)
        self.event_end.record()
        torch.cuda.profiler.stop()
        return self.event_start.elapsed_time(self.event_end)
    

# bucket_sizes = [1, 2, 4, 5, 8, 10 ,16, 20, 25, 40, 50, 80, 100, 125, 200, 250, 400, 500, 1000, 2000]
bucket_sizes = [1, 2, 4, 5, 8, 10 ,16, 20, 25, 40, 50, 80, 100, 125, 200]
# bucket_size = 100
total_size = 2000
actors = [Actor.options(num_gpus=1).remote(total_size) for _ in range(2)]

for bucket_size in bucket_sizes:

    with InputNode() as inp:
        res = []
        start_traces = [actor.start_trace.bind(inp) for actor in actors]
        for _ in range(total_size // bucket_size):
            sends = [actor.send_reduce.bind(inp) for actor in actors]
            res += allreduce.bind(sends)
        end_traces = [actor.end_trace.bind(start) for actor, start in zip(actors, start_traces)]
        dag = MultiOutputNode(end_traces+res)

    compiled_dag = dag.experimental_compile(_overlap_gpu_communication=True)
    ray.get(compiled_dag.execute(bucket_size))

    total_time = 0
    total_time_cpu = 0
    iter = 10
    for _ in range(iter):
        time_start = time.perf_counter()
        times = ray.get(compiled_dag.execute(bucket_size))
        time_end = time.perf_counter()
        total_time += (times[0] + times[1]) / 2
        total_time_cpu += time_end - time_start
    print(f"\n***********************************\nbucket_size: {bucket_size}, avg event time: {total_time / iter:.4f}, avg cpu time: {total_time_cpu * 1000 / iter:.4f}\n")


