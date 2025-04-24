import torch

import ray
from ray.dag import InputNode, MultiOutputNode
from ray.experimental.collective import allreduce


@ray.remote(num_gpus=1)
class Actor:
    def __init__(self):
        self.device = "cuda:0"
        self.compute_pool = [torch.randn(1024, 1024).to(self.device) for _ in range(2)]
        self.transfer_pool = torch.randn(4096, 4096).to(self.device)

        torch.cuda.profiler.start()

    def compute(self, _):
        t = torch.matmul(self.compute_pool[0], self.compute_pool[1])
        for _ in range(250):
            t = torch.matmul(t, self.compute_pool[0])
            t = torch.matmul(t, self.compute_pool[1])

        return 1
    
    def transfer(self, _):
        return self.transfer_pool
    
    def recv_transfer(self, tensors):
        return len(tensors)
    
    def end(self, *args):
        # torch.cuda.synchronize()
        torch.cuda.profiler.stop()
        return 1
    

actors = [Actor.options(num_gpus=1).remote() for _ in range(2)]

with InputNode() as inp:
    computes = [actor.compute.bind(inp) for actor in actors]
    transfers = [actor.transfer.bind(inp) for actor in actors]
    reduced_transfers = allreduce.bind(transfers)
    res = [actor.recv_transfer.bind(reduced_transfer) for actor, reduced_transfer in zip(actors, reduced_transfers)]
    dag = [actor.end.bind(res, compute) for actor, res, compute in zip(actors, res, computes)]
    dag = MultiOutputNode(dag)

compiled_dag = dag.experimental_compile(_overlap_gpu_communication=True)

ref = compiled_dag.execute(None)
ray.get(ref)
# print("##################################################################")

# iter = 5
# for i in range(iter):
#     ref = compiled_dag.execute(None)
#     ray.get(ref)
#     print("##################################################################")
