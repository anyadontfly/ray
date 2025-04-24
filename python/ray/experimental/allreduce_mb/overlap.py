import torch

import ray
from ray.dag import InputNode, MultiOutputNode
from ray.experimental.collective import allreduce


@ray.remote(num_gpus=1)
class Actor:
    def __init__(self):
        torch.cuda.profiler.start()
        self.device = "cuda:0"
        self.compute_tensors = [
            torch.randn(1024, 1024).to(self.device) for _ in range(2)
        ]
        self.transfer_tensor = torch.randn(4096, 4096, 64).to(self.device)
        self.transfer_tensors = (
            torch.randn(4096, 4096, 32).to(self.device),
            torch.randn(4096, 4096, 32).to(self.device),
        )

    def compute(self, _):
        t = torch.matmul(self.compute_tensors[0], self.compute_tensors[1])
        for _ in range(250):
            t = torch.matmul(t, self.compute_tensors[0])
            t = torch.matmul(t, self.compute_tensors[1])
        return 1

    def transfer(self, _):
        return self.transfer_tensors

    def recv_transfer(self, _):
        torch.cuda.profiler.stop()
        return 1


actors = [Actor.options(num_gpus=1).remote() for _ in range(2)]

with InputNode() as inp:
    computes = [actor.compute.bind(inp) for actor in actors]
    transfers = [actor.transfer.bind(inp) for actor in actors]
    reduced_transfers = allreduce.bind(transfers)
    res = [
        actor.recv_transfer.bind(reduced_transfer)
        for actor, reduced_transfer in zip(actors, reduced_transfers)
    ]
    dag = MultiOutputNode(res + computes)

compiled_dag = dag.experimental_compile(_overlap_gpu_communication=True)

iter = 20
for i in range(iter):
    ray.get(compiled_dag.execute(None))
