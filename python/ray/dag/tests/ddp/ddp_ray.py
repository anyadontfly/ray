from typing import Dict, List, Tuple, Optional, Callable, Any

import time

import torch
import torch.nn as nn

import ray
from ray.dag import InputNode, MultiOutputNode
from ray.experimental.collective import allreduce
from ray.air._internal import torch_utils


DEBUG = False

@ray.remote
class MLPActor:
    def __init__(self, fn_init_model: Callable, model_args: Any, batch_size: int = 8):
        torch.cuda.profiler.start()

        torch.manual_seed(42)
        torch.set_default_dtype(torch.float16)
        self.device = torch_utils.get_devices()[0]

        torch.cuda.set_device(self.device)
        torch.cuda.init()

        self.model = fn_init_model(self.device, model_args)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=1)

        self.x = torch.randn(batch_size, model_args[0]).to(self.device)
        self.y = torch.randint(0, 10, (batch_size,)).to(self.device)

        self.intermediates = {}

        if DEBUG:
            self._print_weights()
            self._print_grads()
        

    def forward(self, _) -> None:
        x = self.x
        for idx, layer in enumerate(self.model):
            pred = layer(x)
            if idx < len(self.model) - 1:
                x = pred.detach().requires_grad_(True)
                self.intermediates[idx] = (pred, x)
            else:
                x = pred
                self.intermediates[idx] = (x, x)

    def backward(self, idx: int, _) -> torch.Tensor:
        if idx == len(self.model) - 1:
            loss = self.criterion(self.intermediates[len(self.model) - 1][0], self.y)
            loss.backward()
        else:
            pred, pred_detached = self.intermediates[idx]
            grads = pred_detached.grad
            pred.backward(grads)
        return self.model[idx].weight.grad

    def apply_grads(self, indices: List[int], *grads: torch.Tensor) -> None:
        if len(indices) == 1:
            grads = grads[0]
        for idx, grad in zip(indices, grads):
            self.model[idx].weight.grad = grad

    def update(self, *args: Any) -> None:
        if DEBUG:
            self._print_grads()

        self.optimizer.step()
        self.optimizer.zero_grad()

        if DEBUG:
            self._print_weights()

        torch.cuda.synchronize()
        torch.cuda.profiler.stop()

    def _print_weights(self):
        for name, param in self.model.named_parameters():
            print(f"weight: {name}: {param.data}")
        print("------------------------")

    def _print_grads(self):
        for name, param in self.model.named_parameters():
            print(f"grad: {name}: {param.grad}")
        print("------------------------")


def init_mlp_model(device: str, model_args: Any) -> nn.Module:
    model = []
    for idx in range(len(model_args) - 1):
        layer = nn.Linear(model_args[idx], model_args[idx + 1], bias=False).to(device)
        model.append(layer)
    return nn.Sequential(*model).to(device)


def main():
    num_actors = 2
    num_iters = 5
    time_total = 0

    # Dimensions for the MLP model
    model_args = (2048, 2048, 2048, 2048, 2048, 10)

    # Define buckets for allreduce
    # buckets = [[4], [3], [2], [1], [0]]
    # buckets = [[4, 3], [2, 1], [0]]
    # buckets = [[4, 3, 2], [1, 0]]
    buckets = [[4, 3, 2, 1, 0]]

    actors = [
        MLPActor.options(num_gpus=1).remote(init_mlp_model, model_args) for _ in range(num_actors)
    ]

    with InputNode() as inp:
        # Forward pass
        forward_res = [actor.forward.bind(inp) for actor in actors]
        # Rename
        backward_res = forward_res
        # Gather update node to avoit leaf nodes
        update_res_lst_0 = []
        update_res_lst_1 = []
        for bucket in buckets:
            # Gather backward grads for allreduce
            backward_res_lst_0 = []
            backward_res_lst_1 = []
            for idx in bucket:
                backward_res = [actor.backward.bind(idx, comp) for actor, comp in zip(actors, backward_res)]
                backward_res_lst_0.append(backward_res[0])
                backward_res_lst_1.append(backward_res[1])
            ar_res = allreduce.bind([backward_res_lst_0, backward_res_lst_1])
            # Apply grads to the model
            apply_grads_res = [
                actor.apply_grads.bind(bucket, *grads)
                if isinstance(grads, list)
                else actor.apply_grads.bind(bucket, grads)
                for actor, grads in zip(actors, ar_res)
            ]
            update_res_lst_0.append(apply_grads_res[0])
            update_res_lst_1.append(apply_grads_res[1])

        # Update the model with the applied grads
        update_res = [actor.update.bind(*grad) for actor, grad in zip(actors, [update_res_lst_0, update_res_lst_1])]
        dag = MultiOutputNode(update_res)

    compiled_dag = dag.experimental_compile(_overlap_gpu_communication=True)
    ray.get(compiled_dag.execute(None))

    for _ in range(num_iters):
        start_time = time.perf_counter()
        ray.get(compiled_dag.execute(None))
        end_time = time.perf_counter()
        time_total += (end_time - start_time)
    print(f"Average time per iteration: {time_total / num_iters:.4f} seconds")


if __name__ == "__main__":
    torch.cuda.profiler.start()
    main()
    torch.cuda.profiler.stop()