import ray
from ray.dag import InputNode, MultiOutputNode
from ray.experimental.collective import allreduce
import time
import torch
import argparse

parser = argparse.ArgumentParser(description='Run DAG with configurable bucket size')
parser.add_argument('--bucket-size', type=int, default=10,
                    help='Bucket size for communication (default: 10)')
parser.add_argument('--num-computes', type=int, default=1000,
                    help='Total number of compute operations (default: 1000)')
parser.add_argument('--iterations', type=int, default=5,
                    help='Number of iterations to run (default: 5)')
args = parser.parse_args()

torch.cuda.profiler.start()

@ray.remote(num_gpus=1)
class Actor:
    def __init__(self):
        self.device = "cuda:0"
        self.tensor = torch.ones(1000, 1, device=self.device)
        torch.cuda.profiler.start()

    def compute(self, _):
        res = torch.matmul(self.tensor, self.tensor.T)
        return res
    
    def comm(self, _):
        return self.tensor
    
    def end_trace(self, *args):
        torch.cuda.synchronize()
        torch.cuda.profiler.stop()
        return 1
    
    def recv_tensor(self, *args):
        return 1


actors = [Actor.options(num_gpus=1).remote() for _ in range(2)]

num_computes = args.num_computes
bucket_size = args.bucket_size
iterations = args.iterations

with InputNode() as inp:
    comp_res = [inp, inp]
    ar_res_lst_0 = []
    ar_res_lst_1 = []
    for _ in range(num_computes // bucket_size):
        comp_lst0 = []
        comp_lst1 = []
        for _ in range(bucket_size):
            comp_res = [actor.compute.bind(comp) for actor, comp in zip(actors, comp_res)]
            comp_lst0.append(comp_res[0])
            comp_lst1.append(comp_res[1])
        ar_res = allreduce.bind([comp_lst0, comp_lst1])
        ar_res_lst_0 += ar_res[0]
        ar_res_lst_1 += ar_res[1]

    end_recv = [actor.recv_tensor.bind(*comm) for actor, comm in zip(actors, [ar_res_lst_0, ar_res_lst_1])]
    end_trace = [actor.end_trace.bind(comp) for actor, comp in zip(actors, comp_res)]
    dag = MultiOutputNode(end_recv+end_trace)

compiled_dag = dag.experimental_compile(_overlap_gpu_communication=True)
ray.get(compiled_dag.execute(None))

total_time = 0
for i in range(iterations):
    start_time = time.perf_counter()
    ray.get(compiled_dag.execute(None))
    end_time = time.perf_counter()
    iter_time = end_time - start_time
    total_time += iter_time

avg_time = total_time / iterations

print(f"Bucket size: {bucket_size}, Average time: {avg_time:.4f} seconds")

torch.cuda.profiler.stop()