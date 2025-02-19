import logging
import argparse
from ...core.llama.model import (
    TransformerMP,
    LLAMA_1B,
    LLAMA_3B,
    LLAMA_8B,
    SMALL,
)

import torch


logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="[%(levelname)s %(filename)s:%(lineno)d %(funcName)s] %(message)s",
)

def main(model_size) -> None:
    if model_size == 1:
        model_args = LLAMA_1B
    elif model_size == 3:
        model_args = LLAMA_3B
    elif model_size == 8:
        model_args = LLAMA_8B
    else:
        model_args = SMALL

    device = "cuda:0"

    model = TransformerMP(model_args).to(device)

    vocab_size = model_args.vocab_size
    seq_length = model_args.max_seq_len
    batch_size = 32

    input_tensor = torch.randint(0, vocab_size, (batch_size, seq_length)).to(device)
    target_tensor = torch.randn(batch_size, seq_length, vocab_size).to(device)

    num_epochs = 6
    for epoch in range(num_epochs):
        intermediates = []

        model.bucket_params[0].x = input_tensor
        model.bucket_params[-1].y = target_tensor

        args = model.pre_forward(input_tensor, 0)
        input = model.bucket_params[0].x
        for i, bparam in enumerate(model.bucket_params):
            pred = bparam.forward(input, *args)
            if i < len(model.bucket_params) - 1:
                input = pred.detach().requires_grad_(True)
            else:
                input = pred
            intermediates.append((pred, input))

        for i, bparam in reversed(list(enumerate(model.bucket_params))):
            if i == len(model.bucket_params) - 1:
                loss = bparam.criterion(
                    intermediates[i][0],
                    bparam.y,
                )
                loss_epoch = loss
                pred = None
                grad = None
            else:
                loss = None
                pred, input = intermediates[i]
                grad = input.grad
                intermediates[i] = (None, None)
            grads = bparam.backward(
                loss=loss,
                pred=pred,
                grad=grad,
            )
            if i == 0:
                logger.debug(f"Gradient of first attention layer: {bparam.layers[1].attention.wq.weight.grad}, shape: {bparam.layers[1].attention.wq.weight.shape}")
            bparam.update(grads, True)
            if i == 0:
                logger.debug(f"Params of first attention layer after step: {bparam.layers[1].attention.wq.weight}")

        logger.info(f"Epoch {epoch}, loss: {loss_epoch.item()}")


if __name__ == "__main__":
    torch.manual_seed(42)

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-size",
        type=int,
        default=1,
        choices=[1, 3, 8, 0],
    )

    args = vars(parser.parse_args())
    main(args["model_size"])
