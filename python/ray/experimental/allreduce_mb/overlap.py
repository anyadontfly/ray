import time

import torch

import cupy

import ray
from ray.dag import InputNode, MultiOutputNode
from ray.experimental.collective import allreduce
from ray.experimental.channel import ChannelContext


@ray.remote(num_gpus=1, runtime_env={"nsight": "default"})
class Actor:
    def __init__(self):
        self.device = "cuda:0"
        self.compute_pool = [torch.randn(4096, 4096).to(self.device) for _ in range(2)]
        self.transfer_pool = torch.randn(4096, 4096, 128).to(self.device)

    def compute(self, _):
        ctx = ChannelContext.get_current()

        print(f"compute stream: {torch.cuda.current_stream()}")

        event_compute_start = torch.cuda.Event(enable_timing=True)
        event_compute_start.record(torch.cuda.current_stream())
        ctx._cuda_events["compute"].append(event_compute_start)

        t = torch.matmul(self.compute_pool[0], self.compute_pool[1])
        for _ in range(250):
            t = torch.matmul(t, self.compute_pool[0])
            t = torch.matmul(t, self.compute_pool[1])

        event_compute_end = torch.cuda.Event(enable_timing=True)
        event_compute_end.record(torch.cuda.current_stream())
        ctx._cuda_events["compute"].append(event_compute_end)
        
        return 1
    
    def transfer(self, _):
        return self.transfer_pool
    
    def recv_transfer(self, tensors):
        return len(tensors)
    
    def end(self, *args):
        torch.cuda.synchronize()
        ChannelContext.get_current().conclude_time()
        return 1
    

actors = [Actor.options(num_gpus=1).remote() for _ in range(2)]

with InputNode() as inp:
    computes = [actor.compute.bind(inp) for actor in actors]
    transfers = [actor.transfer.bind(inp) for actor in actors]
    reduced_transfers = allreduce.bind(transfers)
    res = [actor.recv_transfer.bind(reduced_transfer) for actor, reduced_transfer in zip(actors, reduced_transfers)]
    dag = [actor.end.bind(res, compute) for actor, res, compute in zip(actors, res, computes)]
    dag = MultiOutputNode(dag)

compiled_dag = dag.experimental_compile(_overlap_gpu_communication=False)

ref = compiled_dag.execute(None)
ray.get(ref)

# times = []
# avg = 0
# iter = 100
# for i in range(iter):
#     start = time.perf_counter()
#     ref = compiled_dag.execute(None)
#     end = time.perf_counter()
#     avg += (end - start)
#     times.append((end - start) * 1000)
#     ray.get(ref)
