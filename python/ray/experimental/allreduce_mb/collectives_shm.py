import torch

import ray
from ray.dag import InputNode, MultiOutputNode
from ray.experimental.collective import allreduce


torch.cuda.profiler.start()

@ray.remote(num_gpus=1)
class Actor:
    def __init__(self):
        self.device = "cuda:0"
        self.tensors = tuple([torch.ones(1000, device=self.device) for _ in range(100000)])
        torch.cuda.profiler.start()

    def send_reduce(self, _):
        # return torch.ones(1000*num_tensors, device=self.device)
        # return tuple([torch.ones(1000, device=self.device) for _ in range(num_tensors)])
        return self.tensors
    
    def end_trace(self, _):
        torch.cuda.profiler.stop()
        return 1
    

actors = [Actor.options(num_gpus=1).remote() for _ in range(2)]

bucket_size = 100000
total_num_tensors = 1000000
num_allreduces_per_dag = 10
num_iters = 5

with InputNode() as inp:
    lst = []
    for _ in range(num_allreduces_per_dag):
        sends = [actor.send_reduce.bind(inp) for actor in actors]
        results = allreduce.bind(sends)
        lst += results
    ends = [actor.end_trace.bind(inp) for actor in actors]
    dag = MultiOutputNode(lst+ends)


compiled_dag = dag.experimental_compile(_overlap_gpu_communication=True)
for _ in range(num_iters):
    for _ in range(total_num_tensors // bucket_size // num_allreduces_per_dag):
        ray.get(compiled_dag.execute(bucket_size))

torch.cuda.profiler.stop()
