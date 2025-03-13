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


@ray.remote
class MLPActor:
    def __init__(
        self,
        model_dims: List[int],
        batch_size: Optional[int]=8
    ):
        torch.cuda.init()
        torch.manual_seed(42)
        torch.set_default_dtype(torch.float16)
        self.device = torch_utils.get_devices()[0]
        torch.cuda.init()

        model_layers = []
        for i in range(len(model_dims) - 1):
            model_layers.append(nn.Linear(model_dims[i], model_dims[i+1], bias=False).to(self.device))
        self.input = torch.randn(batch_size, model_dims[0]).to(self.device)
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

    def forward(self, _) -> torch.Tensor:
        x = self.input
        for layer in self.model:
            x = layer(x)
            self.intermediates.append(x)
        return x
        
    def backward(self, layers_idx: List[int], t: Optional[torch.Tensor]=None) -> Tuple[torch.Tensor, ...]:
        grads = []

        if layers_idx[0] == len(self.model) - 1:
            output = self.intermediates[-1]
            loss = self.criterion(output, t)
            self.grad_output = torch.autograd.grad(loss, output, retain_graph=True)[0].detach()

        for idx in layers_idx:
            layer = self.model[idx]
            if idx == 0:
                prev_output = self.input
            else:
                prev_output = self.intermediates[idx-1]

            grad = torch.autograd.grad(self.intermediates[idx], [layer.weight], grad_outputs=self.grad_output, retain_graph=True if idx > 0 else False)[0]
            grads.append(grad)

            if idx > 0:
                self.grad_output = torch.autograd.grad(self.intermediates[idx], prev_output, grad_outputs=self.grad_output, retain_graph=True)[0].detach()
        
        return tuple(grads)
    
    def apply_grads(self, layers_idx: List[int], grads: Tuple[torch.Tensor, ...]) -> None:
        for layer_idx, grad in zip(layers_idx, grads):
            print(grad.shape)
            print(layer_idx)
            self.model[layer_idx].weight.grad = grad

        logger = logging.getLogger(__name__)
        # logger.warning(f"Parameters after backward on" + str(layers_idx) + "\n"
        #                + "\n".join(
        #         "{}.grad: {}".format(name, param.grad) for name, param in self.model.named_parameters()))
    
    def update(self, _) -> int:
        self.optimizer.step()
        self.optimizer.zero_grad()
        logger = logging.getLogger(__name__)
        # logger.warning(f"Parameters after update: \n" + "\n".join(
        #         "{}.grad: {}".format(name, param.grad) for name, param in self.model.named_parameters()))
        return 0

def generate_bucket(model_dims, bucket_size):
    layers_size = []
    bucket = []
    curr_bucket = []
    curr_size = 0

    for i in range(len(model_dims) - 1):
        layers_size.append((i, model_dims[i] * model_dims[i+1] * 4 / 1024**2))

    for i, size in layers_size[::-1]:
        if size > bucket_size:
            raise ValueError(f"layer size {size} larger than bucket size")
        if curr_size + size <= bucket_size:
            curr_bucket.append(i)
            curr_size += size
        else:
            bucket.append(curr_bucket)
            curr_bucket = [i]
            curr_size = size
    
    if curr_bucket:
        bucket.append(curr_bucket)

    return bucket

def main():
    num_actors = 2
    model_dims = [1024, 514, 128, 64, 64]
    bucket = generate_bucket(model_dims, 2.1)
    
    logger = logging.getLogger(__name__)
    logger.warning(f"bucket: {bucket}")
    
    models = [MLPActor.options(num_gpus=1).remote(model_dims) for _ in range(num_actors)]
    
    with InputNode() as inp:
        fw_res = [model.forward.bind(inp) for model in models]
        bw_res = fw_res
        for layers_idx in bucket:
            grads = [model.backward.bind(layers_idx, res) for model, res in zip(models, bw_res)]
            grads = allreduce.bind(grads)
            bw_res = [model.apply_grads.bind(layers_idx, grad) for model, grad in zip(models, grads)]
        dag = MultiOutputNode([model.update.bind(res) for model, res in zip(models, bw_res)])

    compiled_dag = dag.experimental_compile()
    ray.get(compiled_dag.execute(None))


if __name__ == "__main__":
    main()
