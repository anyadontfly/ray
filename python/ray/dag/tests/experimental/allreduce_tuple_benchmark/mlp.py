import logging
import time
from collections import defaultdict
from typing import Any, Dict, List, Union, Tuple

import torch
import torch.nn as nn

import ray
from ray.air._internal import torch_utils


@ray.remote
class MLPActor:
    def __init__(
        self,
        rank: int,
        model_dims: str,
    ):
        model_dims = list(map(int, model_dims.split('-')))
        model_layers = []
        for i in range(len(model_dims) - 1):
            model_layers.append(nn.Linear(model_dims[i], model_dims[i + 1]))

        self.model = nn.Sequential(*model_layers)
        self.criteria = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=0.01)

        self.rank = rank
        self.device = torch_utils.get_devices()[0]

        logger = logging.getLogger(__name__)
        for layer in self.model:
            logger.debug(f"Model size: {layer.numel() * layer.element_size() / 1024**2 :.2f} MiB")

        self.time: Dict[str, float] = {}
        self.intermediates: List[torch.Tensor, torch.Tensor] = []

    def forward_and_compute_loss(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        for layer in self.model:
            x = layer(x)
        loss = self.criteria(x, y)
        self.optimizer.zero_grad()
        return loss
        
    def backward(self, loss: torch.Tensor, bucket_size: float) -> Union[torch.Tensor, Tuple[torch.Tensor, ...]]:
        # backward subset of layers that can fit in given bucket size
        start = time.time()
        loss.backward()
        end = time.time()
        self.time["backward"] = end - start
        # get gradients
        grads = []
        for layer in self.model:
            grads.append(layer.weight.grad)

    def update(self, grads_cat: torch.Tensor, grads_passed: bool, idx: int) -> None:
        
