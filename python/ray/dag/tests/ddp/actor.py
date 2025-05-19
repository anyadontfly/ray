import ray

import torch

from .model import Transformer, TransformerBlock, LLAMA_1B


# def traverse_and_apply(module, fn, *args):
#     for _, submodule in module.named_children():
#         if isinstance(submodule, torch.nn.ModuleList):
#             res = traverse_and_apply(submodule, fn, *args)
#         else:
#             fn(submodule, *args)
#         idx += 1
#     return res


@ray.remote(num_gpus=1)
class Llama3Actor:
    def __init__(self):
        self.device = "cuda:0"
        self.model = Transformer(LLAMA_1B).to(self.device)
        self.intermediates = {}

        self.cretirion = torch.nn.CrossEntropyLoss()
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=1e-4, weight_decay=1e-2
        )

        self.num_layers = 0
        for _, submodule in self.model.named_children():
            if isinstance(submodule, torch.nn.ModuleList):
                num_layers += len(submodule)
            else:
                num_layers += 1

    

    def forward(self, x):
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
                else:
                    if isinstance(submodule, TransformerBlock):
                        x = submodule(x, *forward_args).detach().requires_grad_(True)
                    else:
                        x = submodule(x).detach().requires_grad_(True)
                self.intermediates[idx] = x
                idx += 1
            return x, idx
            
        return forward_and_detach(self.model, x, 0, 0, freq_cis, mask)[0]

    def backward(self, x, idx):
        
        def find_layer(module, target_idx, idx):
            for _, submodule in module.named_children():
                if isinstance(submodule, torch.nn.ModuleList):
                    res, idx = find_layer(submodule, target_idx, idx)
                    if res is not None:
                        return res
                else:
                    if idx == target_idx:
                        return submodule, idx
                idx += 1
            return None, idx
        
        layer = find_layer(self.model, idx, 0)[0]

        if idx == self.num_layers - 1:
            loss = self.cretirion(self.intermediates[idx], x)
            loss.backward()
        else:
            pass