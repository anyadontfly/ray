import os
import time

import ray
from ray.dag import InputNode, MultiOutputNode
from ray.experimental.collective import allreduce

import torch
import torch.distributed as dist

@ray.remote(num_gpus=1)
class Actor:
    def __init__(self, rank):
        self.device = "cuda:0"
        self.rank = rank
        os.environ['MASTER_ADDR'] = '127.0.0.1'
        os.environ['MASTER_PORT'] = '29500'

        dist.init_process_group(
            backend='nccl',
            rank=rank,
            world_size=2,
        )

    def start(self, _):
        torch.cuda.profiler.start()
        return 1

    def all_reduce(self, _):
        dist.all_reduce(torch.randn(1000*100, device=self.device), op=dist.ReduceOp.SUM)
        return 
    
    def end(self, *args):
        torch.cuda.profiler.stop()
        return 1



actors = [Actor.options(num_gpus=1).remote(i) for i in range(2)]

with InputNode() as inp:
    computes = []
    starts = [actor.start.bind(inp) for actor in actors]
    for _ in range(100):
        computes += [actor.all_reduce.bind(start) for actor, start in zip(actors, starts)]
    ends = [actor.end.bind(start) for actor, start in zip(actors, starts)]
    dag = MultiOutputNode(ends+computes)

t1 = time.perf_counter()
compiled_dag = dag.experimental_compile(_overlap_gpu_communication=True)
t2 = time.perf_counter()
ray.get(compiled_dag.execute(None))
t3 = time.perf_counter()
print(f"Compilation time: {t2 - t1:.4f} seconds")
print(f"Execution time: {t3 - t2:.4f} seconds")

