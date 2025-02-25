import logging
from ...core.deepseekv3.model import (
    TransformerMP,
    SMALL,
)

import torch


logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="[%(levelname)s %(filename)s:%(lineno)d %(funcName)s] %(message)s",
)

def main() -> None:
    torch.set_default_dtype(torch.bfloat16)
    model_args = SMALL
    device = "cuda:0"

    model = TransformerMP(model_args).to(device)

    vocab_size = model_args.vocab_size
    seq_len = model_args.max_seq_len
    batch_size = 2

    input_tensor = torch.randint(0, model_args.vocab_size, (batch_size, seq_len)).to(device)
    target_tensor = torch.randn(batch_size, seq_len, vocab_size).to(device)

    args = model.pre_forward(input_tensor, 0)
    input = input_tensor
    for i, bparam in enumerate(model.bucket_params):
        input = bparam.forward(input, *args)
    print(input.shape)

    # num_epochs = 6
    # for epoch in range(num_epochs):
    #     intermediates = []

    #     model.bucket_params[0].x = input_tensor
    #     model.bucket_params[-1].y = target_tensor

    #     args = model.pre_forward(input_tensor, 0)
    #     input = model.bucket_params[0].x
    #     for i, bparam in enumerate(model.bucket_params):
    #         pred = bparam.forward(input, *args)
    #         if i < len(model.bucket_params) - 1:
    #             input = pred.detach().requires_grad_(True)
    #         else:
    #             input = pred
    #         intermediates.append((pred, input))

    #     for i, bparam in reversed(list(enumerate(model.bucket_params))):
    #         if i == len(model.bucket_params) - 1:
    #             loss = bparam.criterion(
    #                 intermediates[i][0],
    #                 bparam.y,
    #             )
    #             loss_epoch = loss
    #             pred = None
    #             grad = None
    #         else:
    #             loss = None
    #             pred, input = intermediates[i]
    #             grad = input.grad
    #             intermediates[i] = (None, None)
    #         grads = bparam.backward(
    #             loss=loss,
    #             pred=pred,
    #             grad=grad,
    #         )
    #         if i == 0:
    #             logger.debug(f"Gradient of first attention layer: {bparam.layers[1].attention.wq.weight.grad}, shape: {bparam.layers[1].attention.wq.weight.shape}")
    #         bparam.update(grads, True)
    #         if i == 0:
    #             logger.debug(f"Params of first attention layer after step: {bparam.layers[1].attention.wq.weight}")

    #     logger.info(f"Epoch {epoch}, loss: {loss_epoch.item()}")


if __name__ == "__main__":
    torch.manual_seed(42)

    main()
