import os
import time
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
import torch.multiprocessing as mp

from torch.nn.parallel import DistributedDataParallel as DDP

# On Windows platform, the torch.distributed package only
# supports Gloo backend, FileStore and TcpStore.
# For FileStore, set init_method parameter in init_process_group
# to a local file. Example as follow:
# init_method="file:///f:/libtmp/some_file"
# dist.init_process_group(
#    "gloo",
#    rank=rank,
#    init_method=init_method,
#    world_size=world_size)
# For TcpStore, same way as on Linux.

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'

    # initialize the process group
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()

def demo_basic(rank, world_size):
    setup(rank, world_size)

    # Dimensions for the MLP model
    model_args = (2048, 2048, 2048, 2048, 2048, 10)

    num_iters = 5
    time_total = 0

    def init_mlp_model(rank, model_args):
        model = []
        for i in range(len(model_args) - 1):
            layer = nn.Linear(model_args[i], model_args[i + 1], bias=False).to(rank)
            model.append(layer)
        return nn.Sequential(*model).to(rank)

    # create model and move it to GPU with id rank
    model = init_mlp_model(rank, model_args)
    ddp_model = DDP(model, device_ids=[rank])

    batch_size = 8

    x = torch.randn(batch_size, model_args[0]).to(rank)
    y = torch.randint(0, 10, (batch_size,)).to(rank)

    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.SGD(ddp_model.parameters(), lr=1)

    for _ in range(num_iters):
        time_start = time.perf_counter()
        outputs = ddp_model(x)
        loss_fn(outputs, y).backward()
        optimizer.step()
        optimizer.zero_grad()
        time_end = time.perf_counter()
        time_total += (time_end - time_start)

    print(f"Rank {rank}, Average time per iteration: {time_total / num_iters:.4f} seconds")

    cleanup()


def run_demo(demo_fn, world_size):
    mp.spawn(demo_fn,
             args=(world_size,),
             nprocs=world_size,
             join=True)
    
if __name__ == "__main__":
    run_demo(demo_basic, 2)