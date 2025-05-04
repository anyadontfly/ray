import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import os


def run(rank, world_size):
    # Set up the environment variables for each process
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '29500'
    
    dist.init_process_group(
        backend='nccl',
        rank=rank,
        world_size=world_size,
    )
    
    torch.cuda.set_device(rank)
    device = torch.device(f"cuda:{rank}")
    
    for _ in range(10):
        dist.all_reduce(torch.ones(1000*1000).to(device), op=dist.ReduceOp.SUM)
    
    dist.destroy_process_group()

def main():
    world_size = 2
    mp.spawn(run, args=(world_size,), nprocs=world_size, join=True)

if __name__ == "__main__":
    main()
