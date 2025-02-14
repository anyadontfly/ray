import logging
import os
import time
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP

from ...core.common import log_elapses_to_csv, secs_to_micros
from ...core.config import parse_args
from ...core.linear.actor import BucketParameter

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.WARNING,
    format="[%(levelname)s %(filename)s:%(lineno)d %(funcName)s] %(message)s",
)
logger.info("Welcome to Downton Abbey!")


def run_torch_ddp(
    args: Dict[str, Any]
) -> Tuple[Optional[List[List[torch.Tensor]]], int]:
    num_gpus = torch.cuda.device_count()
    assert num_gpus >= args["num_actors"]
    world_size = args["num_actors"]

    mp.set_start_method("spawn", force=True)

    # Use a multiprocessing manager to share data across devices.
    with mp.Manager() as manager:
        ranks_to_elapses = manager.dict()

        mp.spawn(
            spwan_torch_ddp,
            args=(world_size, ranks_to_elapses, args),
            nprocs=world_size,
            join=True,
        )

        ranks_to_elapses_list = list(ranks_to_elapses[i] for i in range(world_size))

    output_path = args["output_path"]
    latency_prefix = args["latency_prefix"]
    metrics = [
        "total",
        "fw.total",
        "loss.compute",
        "bw.bw_ar",
        "bw.update",
        "barrier",
    ]
    log_elapses_to_csv(
        ranks_to_elapses_list,
        output_path,
        latency_prefix,
        metrics,
    )


def spwan_torch_ddp(
    rank: int,
    world_size: int,
    ranks_to_elapses: Dict[int, int],
    args: Dict[str, Any],
) -> None:
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "8888"

    try:
        logger = logging.getLogger(__name__)

        # Initialize the process group.
        dist.init_process_group("nccl", rank=rank, world_size=world_size)

        # Create model on the device.
        device = f"cuda:{rank}"
        model = BucketParameter(
            args["layer_size"],
            args["num_layers"],
            device,
        )
        size_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
        logger.warning(f"Model size: {size_bytes / 1024 / 1024} MB")

        torch.manual_seed(998244353)
        model.init_weights()
        model = model.to(model.device)
        ddp_model = DDP(model, device_ids=[rank])

        elapses = defaultdict(list)

        for epoch in range(args["num_epochs"]):
            model.x = torch.randn(
                1,
                model.layer_size,
                requires_grad=True,
            ).to(
                model.device,
            )
            model.y = torch.randn(
                1,
                model.layer_size,
            ).to(
                model.device,
            )

            if rank == 0:
                logger.info(f"epoch: {epoch}")
                logger.info(f"input: {model.x}")
                logger.info(f"target: {model.x}")

            dist.barrier()
            start = time.perf_counter()

            forward_start = time.perf_counter()
            pred = ddp_model(model.x)
            forward_end = time.perf_counter()

            loss_compute_start = time.perf_counter()
            loss = model.criterion(pred, model.y)
            loss_compute_end = time.perf_counter()

            backward_start = time.perf_counter()
            loss.backward()
            backward_end = time.perf_counter()

            update_start = time.perf_counter()
            model.optimizer.step()
            model.optimizer.zero_grad()
            update_end = time.perf_counter()

            barrier_start = time.perf_counter()
            dist.barrier()
            end = time.perf_counter()

            if rank == 0:
                weights = model.fetch_weights()
                for i, weight in enumerate(weights):
                    logger.info(f"layer: {i}, weight: {weight}")

            total = end - start

            def log(key: str, elapse: float):
                elapses[key].append(secs_to_micros(elapse))
                logger.warning(
                    f"rank: {rank}, {key} elapse: {secs_to_micros(elapse)} us, percent: {round(elapse / total * 100, 1)}%"
                )

            log("total", total)
            log("fw.total", forward_end - forward_start)
            log("loss.compute", loss_compute_end - loss_compute_start)
            log("bw.bw_ar", backward_end - backward_start)
            log("bw.update", update_end - update_start)
            log("barrier", end - barrier_start)
    finally:
        # Destroy the process group.
        dist.destroy_process_group()

    ranks_to_elapses[rank] = elapses

    save_model = args.get("save_model", False)
    if save_model:
        if rank == 0:
            model_file = f"{args['model_prefix']}.log"
            with open(model_file, "w") as f:
                weights = model.fetch_weights()
                for weight in weights:
                    f.write(f"{weight}\n")
        model_file = f"{args['model_prefix']}_{rank}.log"
        with open(model_file, "w") as f:
            weights = model.fetch_weights()
            for weight in weights:
                f.write(f"{weight.cpu()}\n")


if __name__ == "__main__":
    args = parse_args()
    run_torch_ddp(args)
