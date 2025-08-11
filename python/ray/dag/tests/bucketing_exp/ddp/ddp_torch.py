import os
import time

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
import torch.cuda.nvtx as nvtx

from torch.nn.parallel import DistributedDataParallel as DDP

from llama3 import Transformer, LLAMA_1B


def demo_basic():
    rank = int(os.environ["LOCAL_RANK"])
    device = torch.device(f"cuda:{rank}")
    torch.distributed.init_process_group(backend="nccl", device_id=device)
    torch.set_default_dtype(torch.bfloat16)

    num_iters = 10
    batch_size = 2
    seq_len = 128

    # create model and move it to GPU with id rank
    model = Transformer(LLAMA_1B).to(rank)
    ddp_model = DDP(model, device_ids=[rank], bucket_cap_mb=365)

    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.SGD(ddp_model.parameters(), lr=0.001)

    x = torch.randint(0, LLAMA_1B.vocab_size, (batch_size, seq_len)).to(rank)
    y = torch.randn(batch_size, seq_len, LLAMA_1B.vocab_size).to(rank)

    for _ in range(num_iters):
        event_start = torch.cuda.Event(enable_timing=True)
        event_end = torch.cuda.Event(enable_timing=True)

        nvtx.mark("start")
        event_start.record()
        # forward pass
        outputs = ddp_model(x, 0)
        loss = loss_fn(outputs, y)

        # backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        nvtx.mark("end")
        event_end.record()
        event_end.synchronize()
        elapsed_time = event_start.elapsed_time(event_end)
        if rank == 0:
            print(f"Time taken for forward and backward: {elapsed_time} ms")
        time.sleep(0.5)

    dist.destroy_process_group()


if __name__ == "__main__":
    demo_basic()