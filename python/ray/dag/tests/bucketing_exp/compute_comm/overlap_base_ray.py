import ray
from ray.dag import InputNode, MultiOutputNode
from ray.experimental.collective import allreduce
import time
import torch


USE_PROFILER = True
USE_EVENT = False

if USE_PROFILER:
    torch.cuda.profiler.start()

@ray.remote(num_gpus=1)
class Actor:
    def __init__(self):
        self.device = "cuda:0"
        self.tensor = torch.ones(100000, 1000, device=self.device)
        self.res = torch.ones(1000, 1, device=self.device)
        if USE_PROFILER:
            torch.cuda.profiler.start()

    def ret(self, _):
        return None

    def compute(self, _):
        torch.matmul(self.tensor, self.res)
        return 1
    
    def comm(self, _):
        return self.res
    
    def end_trace(self, *args):
        torch.cuda.synchronize()
        if USE_PROFILER:
            torch.cuda.profiler.stop()
        return args
    
    def recv_tensor(self, *args):
        return 1


actors = [Actor.options(num_gpus=1).remote() for _ in range(2)]

with InputNode() as inp:
    rets = [actor.ret.bind(inp) for actor in actors]
    comp_res = [actor.compute.bind(ret) for actor, ret in zip(actors, rets)]
    comm_res = [actor.comm.bind(ret) for actor, ret in zip(actors, rets)]
    ar_res = allreduce.bind(comm_res)
    end_trace = [actor.end_trace.bind(comm, comp) for actor, comm, comp in zip(actors, ar_res, comp_res)]
    dag = MultiOutputNode(end_trace)

compiled_dag = dag.experimental_compile(_overlap_gpu_communication=True)
ray.get(compiled_dag.execute(None))

total_time = 0
total_time_event = 0
torch.cuda.synchronize()
for i in range(5):
    start_time = time.perf_counter()
    if USE_EVENT:
        start_event = torch.cuda.Event(enable_timing=True)
        start_event.record()

    ray.get(compiled_dag.execute(None))

    end_time = time.perf_counter()
    if USE_EVENT:
        end_event = torch.cuda.Event(enable_timing=True)
        end_event.record()

    iter_time = end_time - start_time
    total_time += iter_time

    if USE_EVENT:
        torch.cuda.synchronize()
        total_time_event += start_event.elapsed_time(end_event)

avg_time = total_time / 5
if USE_EVENT:
    avg_time_event = total_time_event / 5

print(f"!!!!!!!!!!!!! Average execution time over 5 runs: {avg_time:.4f} seconds")
if USE_EVENT:
    print(f"!!!!!!!!!!!!! Average time by event over 5 runs: {avg_time_event:.4f} ms")

if USE_PROFILER:
    torch.cuda.profiler.stop()