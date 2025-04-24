import logging
from typing import Dict, List, Tuple, Optional

import torch
import torch.nn as nn

import ray
from ray.dag import InputNode, MultiOutputNode
from ray.experimental.collective import allreduce
from ray.air._internal import torch_utils

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.WARNING, format="%(asctime)s - %(levelname)s - %(message)s")


class Bucket:
    def __init__(self, id: int):
        self.id = id
        self.layer_indices = []
        self._is_last_bucket = False

    def add_layer(self, layer_idx: List[int]):
        self.layer_indices.append(layer_idx)

    @property
    def is_empty(self):
        return len(self.layer_indices) == 0
    
    @property
    def is_last_baucket(self):
        return self._is_last_bucket
    
    def __str__(self):
        return str(self.layer_indices)


@ray.remote
class MLPActor:
    def __init__(
        self,
        model_dims: List[int],
        batch_size: Optional[int]=8
    ):
        torch.manual_seed(42)
        torch.set_default_dtype(torch.float16)
        self.device = torch_utils.get_devices()[0]

        model_layers = []
        for i in range(len(model_dims) - 1):
            model_layers.append(nn.Linear(model_dims[i], model_dims[i+1], bias=False).to(self.device))
        self.x = torch.randn(batch_size, model_dims[0]).to(self.device)
        self.y = torch.randint(0, model_dims[-1], (batch_size,)).to(self.device)
        self.intermediates: List[torch.Tensor] = []

        self.model = nn.Sequential(*model_layers)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=0.01)

        logger = logging.getLogger(__name__)
        for layer in self.model:
            logger.warning(f"Layer size: {layer.weight.numel() * layer.weight.element_size() / 1024**2 :.2f} MiB")

        self.time: Dict[str, float] = {}

        # logger.warning("initial parameters:\n" + "\n".join(
        #         "{}.grad: {}".format(name, param.grad) for name, param in self.model.named_parameters()))

    def forward(self, buckets) -> torch.Tensor:
        x = self.x
        for i, bucket in enumerate(buckets):
            for i, layer_idx in enumerate(bucket.layer_indices):
                x = self.model[layer_idx](x)
                if layer_idx == len(buckets) - 1 or i == len(bucket.layer_indices) - 1:
                    self.intermediates.append(x)
                else:
                    self.intermediates.append(x.detach().requires_grad_(True))

        logger = logging.getLogger(__name__)
        logger.warning(f"inter shapes: {[inter.shape for inter in self.intermediates]}")
        return x
        
    def backward(self, bucket: Bucket, t: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        logger = logging.getLogger(__name__)
        logger.warning(f"backward: {str(bucket)}")

        grads = []

        if bucket.is_last_baucket:
            logger.warning(f"output: {t.shape}")
            logger.warning(f"y: {self.y.shape}")
            loss = self.criterion(t, self.y)
            loss.backward()
        else:
            last_layer_idx_in_bucket = bucket.layer_indices[-1]
            logger.warning(f"{last_layer_idx_in_bucket}:{self.intermediates[last_layer_idx_in_bucket].shape}, {last_layer_idx_in_bucket+1}:{self.intermediates[last_layer_idx_in_bucket+1].shape}")
            self.intermediates[last_layer_idx_in_bucket].backward(self.intermediates[last_layer_idx_in_bucket])

        for layer_idx in bucket.layer_indices:
            grads.append(self.model[layer_idx].weight.grad)

        logger.warning(f"backward res: {[grad.shape for grad in grads]}")

        return tuple(grads)
    
    def apply_grads(self, bucket: Bucket, grads: Tuple[torch.Tensor, ...]) -> None:
        for layer_idx, grad in zip(bucket.layer_indices, grads):
            self.model[layer_idx].weight.grad = grad

        # logger = logging.getLogger(__name__)
        # logger.warning(f"Parameters after backward on" + str(bucket) + "\n"
        #                + "\n".join(
        #         "{}.grad: {}".format(name, param.grad) for name, param in self.model.named_parameters()))
        return grads[0]
    
    def update(self, _) -> int:
        self.optimizer.step()
        self.optimizer.zero_grad()
        # logger = logging.getLogger(__name__)
        # logger.warning(f"Parameters after update: \n" + "\n".join(
        #         "{}.grad: {}".format(name, param.grad) for name, param in self.model.named_parameters()))
        return 0

def generate_bucket(model_dims, bucket_size):
    layers_size = []
    buckets = []

    bucket_id = 0
    curr_bucket = Bucket(bucket_id)
    curr_size = 0

    for i in range(len(model_dims) - 1):
        layers_size.append((i, model_dims[i] * model_dims[i+1] * 4 / 1024**2))

    for i, size in layers_size:
        if size > bucket_size:
            raise ValueError(f"layer size {size} larger than bucket size")
        if curr_size + size <= bucket_size:
            curr_bucket.add_layer(i)
            curr_size += size
        else:
            buckets.append(curr_bucket)
            bucket_id += 1
            curr_bucket = Bucket(bucket_id)
            curr_bucket.add_layer(i)
            curr_size = size
    
    if not curr_bucket.is_empty:
        buckets.append(curr_bucket)

    buckets[-1]._is_last_bucket = True
    return buckets

def main():
    num_actors = 2
    model_dims = [1024, 512, 128, 64, 1]
    buckets = generate_bucket(model_dims, 2.1)
    
    logger = logging.getLogger(__name__)
    logger.warning(f"bucket: {[str(bucket) for bucket in buckets]}")
    
    actors = [MLPActor.options(num_gpus=1).remote(model_dims) for _ in range(num_actors)]
    
    with InputNode() as inp:
        fw_res = [actor.forward.bind(inp) for actor in actors]
        bw_res = fw_res
        for bucket in buckets[::-1]:
            actors_grads = [actor.backward.bind(bucket, res) for actor, res in zip(actors, bw_res)]
            actors_grads = allreduce.bind(actors_grads)
            bw_res = [actor.apply_grads.bind(bucket, actor_grads) for actor, actor_grads in zip(actors, actors_grads)]
        dag = MultiOutputNode([actor.update.bind(res) for actor, res in zip(actors, bw_res)])

    compiled_dag = dag.experimental_compile()
    ray.get(compiled_dag.execute(buckets))


if __name__ == "__main__":
    main()