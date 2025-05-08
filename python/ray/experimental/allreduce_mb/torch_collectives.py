import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import os
import time


def run(rank, world_size):
    # Set up the environment variables for each process
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '29500'
    
    dist.init_process_group(
        backend='nccl',
        rank=rank,
        world_size=world_size,
    )

    bucket_size = 1000
    total_num_tensors = 10_000
    
    torch.cuda.set_device(rank)
    device = torch.device(f"cuda:{rank}")
    
    for _ in range(10):
        for _ in range(total_num_tensors // bucket_size):
            tensors = [torch.ones(1000, device=device) for _ in range(bucket_size)]
            flat_buf = torch.nn.utils.parameters_to_vector(tensors)
            dist.all_reduce(flat_buf, op=dist.ReduceOp.SUM)
    
    dist.destroy_process_group()

def main():
    world_size = 2
    mp.spawn(run, args=(world_size,), nprocs=world_size, join=True)

if __name__ == "__main__":
    TIMING = False
    if TIMING:
        t1 = time.perf_counter()
    main()
    if TIMING:
        t2 = time.perf_counter()
        print(f"Execution time: {t2 - t1:.4f} seconds")
