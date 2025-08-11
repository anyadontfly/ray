from typing import List, Any

import time

import torch
import torch.nn as nn
import torch.cuda.nvtx as nvtx

import ray
from ray.dag import InputNode, MultiOutputNode
from ray.experimental.collective import allreduce
from ray.air._internal import torch_utils

from llama3 import Transformer, TransformerBlock, LLAMA_1B


DEBUG = False

@ray.remote
class Llama3Actor:
    def __init__(self, model_args: Any, batch_size: int = 2):
        torch.cuda.profiler.start()

        torch.manual_seed(42)
        self.device = torch_utils.get_devices()[0]

        torch.set_default_dtype(torch.bfloat16)

        torch.cuda.set_device(self.device)
        torch.cuda.init()

        self.model = Transformer(model_args).to(self.device)
    
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=1)

        seq_len = 128
        vocab_size = 128256

        self.x = torch.randint(0, vocab_size, (batch_size, seq_len)).to(self.device)
        self.y = torch.randn(batch_size, seq_len, vocab_size).to(self.device)

        self.intermediates = {}

        if DEBUG:
            self._print_weights()
            self._print_grads()

        self.num_layers = 0
        for _, submodule in self.model.named_children():
            if isinstance(submodule, torch.nn.ModuleList):
                self.num_layers += len(submodule)
            else:
                self.num_layers += 1

        self.event_start = torch.cuda.Event(enable_timing=True)
        self.event_end = torch.cuda.Event(enable_timing=True)
        
    def forward(self, _) -> None:
        time.sleep(0.5)
        nvtx.mark("start")
        self.event_start.record()

        x = self.x

        _, seqlen = x.shape
        freq_cis = self.model.freqs_cis[0:seqlen].to(self.device)
        mask = torch.full((seqlen, seqlen), float("-inf"), device=self.device)
        mask = torch.triu(mask, diagonal=1)
        mask = torch.hstack(
                    [torch.zeros((seqlen, 0), device=self.device), mask]
                )
        
        def forward_and_detach(module, x, idx, *forward_args):
            for _, submodule in module.named_children():
                if isinstance(submodule, torch.nn.ModuleList):
                    x, idx = forward_and_detach(submodule, x, idx, *forward_args)
                    idx -= 1
                else:
                    if isinstance(submodule, TransformerBlock):
                        # This is a TransformerBlock, so we need to pass the frequency and mask
                        # Since this is not final layer, we need to detach the output
                        pred = submodule(x, *forward_args)
                        x = pred.detach().requires_grad_(True)
                        self.intermediates[idx] = (pred, x)
                    elif idx < self.num_layers - 1:
                        # Not a TransformerBlock, also not final layer, we need to detach the output
                        pred = submodule(x)
                        x = pred.detach().requires_grad_(True)
                        self.intermediates[idx] = (pred, x)
                    else:
                        # Final layer is not a TransformerBlock, we can just pass the output
                        pred = submodule(x)
                        x = pred
                        self.intermediates[idx] = (x, x)
                idx += 1
            return x, idx
            
        forward_and_detach(self.model, x, 0, 0, freq_cis, mask)

    def backward(self, idx: int, _) -> torch.Tensor:
        layer = self._find_layer(self.model, idx, 0)[0]

        if idx == self.num_layers - 1:
            loss = self.criterion(self.intermediates[self.num_layers - 1][0], self.y)
            loss.backward()
        else:
            pred, pred_detached = self.intermediates[idx]
            grads = pred_detached.grad
            pred.backward(grads)

        def gather_grads(block):
            grads = []
            for param in block.parameters():
                if param.grad is not None:
                    grads.append(param.grad.view(-1))
                else:
                    grads.append(torch.zeros_like(param.data).view(-1))
            return torch.cat(grads)
        
        if isinstance(layer, TransformerBlock):
            return gather_grads(layer)
        else:
            return layer.weight.grad

    def apply_grads(self, indices: List[int], *grads: torch.Tensor) -> None:
        if len(indices) == 1:
            grads = grads[0]
        for idx, grad in zip(indices, grads):
            layer = self._find_layer(self.model, idx, 0)[0]

            def set_grads_to_block(block, flat_tensor):
                pointer = 0
                for param in block.parameters():
                    numel = param.numel()
                    grad = flat_tensor[pointer: pointer + numel].view_as(param.data)
                    param.grad = grad
                    pointer += numel

            if isinstance(layer, TransformerBlock):
                set_grads_to_block(layer, grad)
            else:
                layer.weight.grad = grad

    def update(self, *args: Any) -> None:
        if DEBUG:
            self._print_grads()

        self.optimizer.step()
        self.optimizer.zero_grad()

        if DEBUG:
            self._print_weights()

        nvtx.mark("end")
        self.event_end.record()
        self.event_end.synchronize()
        print(f"Time taken for forward and backward: {self.event_start.elapsed_time(self.event_end)} ms")
        torch.cuda.profiler.stop()

    def _find_layer(self, module, target_idx, idx):
        for _, submodule in module.named_children():
            if isinstance(submodule, torch.nn.ModuleList):
                res, idx = self._find_layer(submodule, target_idx, idx)
                idx -= 1
                if res is not None:
                    return res, idx
            else:
                if idx == target_idx:
                    return submodule, idx
            idx += 1
        return None, idx

    def _print_weights(self):
        for name, param in self.model.named_parameters():
            print(f"weight: {name}: {param.data}")
        print("------------------------")

    def _print_grads(self):
        for name, param in self.model.named_parameters():
            print(f"grad: {name}: {param.grad}")
        print("------------------------")


def generate_buckets(bucket_size: int, model_args: Any) -> List[List[int]]:
    """
    bucket_size: in number of layers
    """
    buckets = []
    num_layers = len(model_args) - 1
    for i in range(0, num_layers, bucket_size):
        bucket = list(range(i, min(i + bucket_size, num_layers)))
        buckets.append(bucket)
    # reverse the nested lists and the outer list
    buckets = [bucket[::-1] for bucket in buckets]
    return buckets[::-1]

def run_ddp(actors: Any, Any, buckets):
    
    num_iters = 10
    # time_total = 0

    with InputNode() as inp:
        # Forward pass
        forward_res = [actor.forward.bind(inp) for actor in actors]
        # Rename
        backward_res = forward_res
        # Gather update node to avoit leaf nodes
        update_res_lst_0 = []
        update_res_lst_1 = []
        ar_res_list = []
        for bucket in buckets:
            # Gather backward grads for allreduce
            backward_res_lst_0 = []
            backward_res_lst_1 = []
            for idx in bucket:
                backward_res = [actor.backward.bind(idx, comp) for actor, comp in zip(actors, backward_res)]
                backward_res_lst_0.append(backward_res[0])
                backward_res_lst_1.append(backward_res[1])
            ar_res = allreduce.bind([backward_res_lst_0, backward_res_lst_1])
            ar_res_list.append(ar_res)
        
        for bucket, ar_res in zip(buckets, ar_res_list):
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
        # start_time = time.perf_counter()
        ray.get(compiled_dag.execute(None))
        # end_time = time.perf_counter()
        # time_total += (end_time - start_time)
    # print(f"Average time per iteration: {time_total / num_iters:.4f} seconds\n")


if __name__ == "__main__":
    torch.cuda.profiler.start()

    num_actors = 2

    actors = [
        Llama3Actor.options(num_gpus=1).remote(LLAMA_1B) for _ in range(num_actors)
    ]

    bucketing_schedules = [
        # [[18], [17], [16], [15], [14], [13], [12], [11], [10], [9], [8], [7], [6], [5], [4], [3], [2], [1], [0]],
        # [[18], [17, 16, 15], [14, 13], [12, 11], [10, 9], [8, 7], [6, 5], [4, 3], [2, 1], [0]],
        [[18], [17, 16], [15, 14, 13], [12, 11, 10], [9, 8, 7], [6, 5, 4], [3, 2, 1], [0]],
        # [[18], [17, 16, 15, 14, 13], [12, 11, 10, 9], [8, 7, 6, 5], [4, 3, 2, 1], [0]],
        # [[18], [17, 16], [15, 14, 13, 12, 11], [10, 9, 8, 7, 6], [5, 4, 3, 2, 1], [0]],
        # [[18], [17, 16, 15, 14, 13], [12, 11, 10, 9, 8, 7], [6, 5, 4, 3, 2, 1], [0]],
        # [[18], [17, 16, 15], [14, 13, 12, 11, 10, 9, 8], [7, 6, 5, 4, 3, 2, 1], [0]],
        # [[18], [17, 16, 15, 14, 13, 12, 11, 10, 9], [8, 7, 6, 5, 4, 3, 2, 1], [0]],
        # [[18], [17, 16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1], [0]],
    ]

    for bucketing_schedule in bucketing_schedules:
        run_ddp(actors, LLAMA_1B, bucketing_schedule)
    torch.cuda.profiler.stop()