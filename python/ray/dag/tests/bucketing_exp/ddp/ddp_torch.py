import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import torch.optim as optim

from torch.nn.parallel import DistributedDataParallel as DDP

from .llama3 import Transformer, LLAMA_1B


def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'

    # initialize the process group
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

    torch.set_default_dtype(torch.float16)


def cleanup():
    dist.destroy_process_group()


def demo_basic(rank, world_size):
    setup(rank, world_size)

    batch_size = 2

    # create model and move it to GPU with id rank
    model = Transformer(LLAMA_1B).to(rank)
    ddp_model = DDP(model, device_ids=[rank], bucket_cap_mb=10)

    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.SGD(ddp_model.parameters(), lr=0.001)

    x = torch.randint(0, LLAMA_1B.vocab_size, (batch_size, 128)).to(rank)
    y = torch.randn(batch_size, 128, LLAMA_1B.vocab_size).to(rank)

    for _ in range(10):
        # forward pass
        outputs = ddp_model(x, 0)
        loss = loss_fn(outputs, y)

        # backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    cleanup()


def run_demo(demo_fn, world_size):
    mp.spawn(demo_fn,
             args=(world_size,),
             nprocs=world_size,
             join=True)


if __name__ == "__main__":
    run_demo(demo_basic, 2)