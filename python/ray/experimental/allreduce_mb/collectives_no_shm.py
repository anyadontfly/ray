import time

import torch

import ray
from ray.dag import InputNode, MultiOutputNode
from ray.experimental.collective import allreduce


torch.cuda.profiler.start()

@ray.remote(num_gpus=1)
class Actor:
    def __init__(self):
        self.device = "cuda:0"
        torch.cuda.profiler.start()

    def send_reduce(self, num_tensors):
        # return torch.ones(1000*num_tensors, device=self.device)
        return tuple([torch.ones(1000, device=self.device) for _ in range(num_tensors)])
    
    def recv_reduce(self, _):
        return 1
    
    def end_trace(self, _):
        torch.cuda.profiler.stop()
        return 1
    

actors = [Actor.options(num_gpus=1).remote() for _ in range(2)]

bucket_size = 1000
total_num_tensors = 10_000

with InputNode() as inp:
    lst = []
    for _ in range(total_num_tensors // bucket_size):
        sends = [actor.send_reduce.bind(inp) for actor in actors]
        results = allreduce.bind(sends)
        recvs = [actor.recv_reduce.bind(res) for actor, res in zip(actors, results)]
        lst += recvs
    ends = [actors[0].end_trace.bind(lst[0]), actors[1].end_trace.bind(lst[1])]
    dag = MultiOutputNode(lst+ends)

t1 = time.perf_counter()
compiled_dag = dag.experimental_compile(_overlap_gpu_communication=True)
t2 = time.perf_counter()
ray.get(compiled_dag.execute(bucket_size))
t3 = time.perf_counter()
print(f"Compilation time: {t2 - t1:.4f} seconds")
print(f"Execution time: {t3 - t2:.4f} seconds")

torch.cuda.profiler.stop()
