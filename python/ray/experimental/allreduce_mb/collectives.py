import torch

import ray
from ray.dag import InputNode, MultiOutputNode
from ray.experimental.collective import allreduce


torch.cuda.profiler.start()

@ray.remote(num_gpus=1)
class Actor:
    def __init__(self, num_tensors):
        self.device = "cuda:0"
        self.tensors = tuple([torch.ones(1000, device=self.device) for _ in range(num_tensors)])
        # self.tensors = torch.ones(1000*num_tensors, device=self.device)
        torch.cuda.profiler.start()

    def gen_tensor(self, _):
        return self.tensors
    
    def end_trace(self, *args):
        torch.cuda.profiler.stop()
        return 1
    

bucket_size = 10
num_total_tensors = 10000
num_allreduces_per_dag = 100

num_dags = num_total_tensors // bucket_size // num_allreduces_per_dag
num_iters = 10

actors = [Actor.options(num_gpus=1).remote(bucket_size) for _ in range(2)]

with InputNode() as inp:
    res_lst0 = []
    res_lst1 = []
    tensors = [actor.gen_tensor.bind(inp) for actor in actors]
    for _ in range(num_allreduces_per_dag):
        results = allreduce.bind(tensors)
        res_lst0.append(results[0])
        res_lst1.append(results[1])
    ends = [actor.end_trace.bind(*res_lst) for actor, res_lst in zip(actors, [res_lst0, res_lst1])]
    dag = MultiOutputNode(ends)

compiled_dag = dag.experimental_compile(_overlap_gpu_communication=True)

for _ in range(num_iters):
    for _ in range(num_dags):
        ray.get(compiled_dag.execute(bucket_size))

torch.cuda.profiler.stop()