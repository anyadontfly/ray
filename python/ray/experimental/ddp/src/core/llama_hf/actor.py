import logging
import time
from collections import defaultdict
from typing import Any, Dict, List

import torch

import ray
from ..common import secs_to_micros
from ..generation import LlamaMP

@ray.remote
class LlamaActor:
    def __init__(
        self,
        rank: int,
        num_models: int,
        num_actors: int,
        device: torch.device,
        check_tracing: bool,
    ):
        self.model = LlamaMP()
        self.buckets = [bparam.to(device) for bparam in self.model.bucket_params]

        self.rank = rank
        self.num_models = num_models
        self.num_actors = num_actors
        self.device = device
        self.check_tracing = check_tracing

        logger = logging.getLogger(__name__)
        for model in self.models:
            size_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
            logger.warning(f"Model size: {size_bytes / 1024 / 1024:.2f} MiB")
        self.intermediates: List[torch.Tensor, torch.Tensor] = []

        self.it = 0
        self.time: Dict[str, Any] = {}
        self.elapses: Dict[str, List] = defaultdict(list)

    def init_training(self, batch_size: int) -> None:
        self.models[0].x = torch.full((batch_size, 1000),
            torch.randint(0, 128256, (1,)).item(),
            dtype=torch.long
        ).to(
            self.device,
        )
        self.models[-1].y = torch.randint(
            0,
            1,
            (batch_size, 1000, 128256),
        ).to(
            self.device,
        )

    def forward(self, _) -> None:
        self.update_time("start")
        if self.check_tracing:
            self.update_time("forward_starts")
        self.intermediates = []
        input = self.models[0].x
        for i, model in enumerate(self.models):
            pred = model.forward(input)
            if i < len(self.models) - 1:
                input = pred.detach().requires_grad_(True)
            else:
                input = pred
            self.intermediates.append((pred, input))
        if self.check_tracing:
            self.update_time("forward_ends")

    def backward(self, _, idx: int) -> torch.Tensor:
        if self.check_tracing:
            self.update_time("backward_starts")
        if idx == len(self.models) - 1:
            loss = self.models[idx].criterion(
                self.intermediates[idx][0],
                self.models[idx].y,
            )
            pred = None
            grad = None
        else:
            loss = None
            pred, input = self.intermediates[idx]
            grad = input.grad
            self.intermediates[idx] = (None, None)
        grads = self.models[idx].backward(
            loss=loss,
            pred=pred,
            grad=grad,
        )
        if self.check_tracing:
            self.update_time("backward_ends")
        return grads

    def update(self, grads_cat: torch.Tensor, grads_passed: bool, idx: int) -> None:
        if self.check_tracing:
            self.update_time("update_starts")
        if grads_passed:
            grads_cat /= self.num_actors
        self.models[idx].update(grads_cat, grads_passed)
        if self.check_tracing:
            self.update_time("update_ends")
        if idx == 0:
            self.update_time("end")
        