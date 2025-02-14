import logging
import argparse
from ...core.llama_meta.model import (
    Transformer,
    TransformerMP,
    LLAMA_1B,
    LLAMA_3B,
    LLAMA_8B,
    TRANSFORMER_SMALL,
)

import torch


logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.WARNING,
    format="[%(levelname)s %(filename)s:%(lineno)d %(funcName)s] %(message)s",
)

def main(model_type, model_size) -> None:
    if model_type == "mp":
        logger.warning("Bucket training started!")

        if model_size == 1:
            model = TransformerMP(LLAMA_1B).to("cuda")
        elif model_size == 3:
            model = TransformerMP(LLAMA_3B).to("cuda")
        elif model_size == 8:
            model = TransformerMP(LLAMA_8B).to("cuda")
        else:
            model = TransformerMP(TRANSFORMER_SMALL).to("cuda")

        input_tensor = torch.randint(0, 128256, (1, 2048)).to("cuda")
        target_tensor = torch.randn(1, 2048, 128256, requires_grad=True).to("cuda")

        intermediates = []

        model.bucket_params[0].x = input_tensor
        model.bucket_params[-1].y = target_tensor

        args = model.pre_forward(input_tensor, 0)
        output = model.bucket_params[0].x
        for i, bparams in enumerate(model.bucket_params):
            pred = bparams.forward(output, *args)
            if i < len(model.bucket_params) - 1:
                output = pred.detach().requires_grad_(True)
            else:
                output = pred
            intermediates.append((pred, output))

        logger.warning(f"Output shape: {output.shape}, output: {output}")

        for i in reversed(range(len(model.bucket_params))):
            if i == len(model.bucket_params) - 1:
                loss = model.bucket_params[i].criterion(
                    intermediates[i][0],
                    model.bucket_params[i].y,
                )
                pred = None
                grad = None
            else:
                loss = None
                pred, input = intermediates[i]
                grad = input.grad
                intermediates[i] = (None, None)
            grads = model.bucket_params[i].backward(
                loss=loss,
                pred=pred,
                grad=grad,
            )
            model.bucket_params[i].update(grads, True)

        logger.warning("Bucket training completed!")

    else:
        logger.warning("Normal Transformer training started!")

        if model_size == 1:
            model = Transformer(LLAMA_1B).to("cuda")
        elif model_size == 3:
            model = Transformer(LLAMA_3B).to("cuda")
        elif model_size == 8:
            model = Transformer(LLAMA_8B).to("cuda")
        else:
            model = Transformer(TRANSFORMER_SMALL).to("cuda")

        # Assuming you have a Transformer model and input data
        input_tensor = torch.randint(0, 128256, (1, 2048)).to("cuda")
        target_tensor = torch.randn(1, 2048, 128256, requires_grad=True).to("cuda")

        # Define the optimizer
        optimizer = torch.optim.Adam(model.parameters())

        # Zero the gradients
        optimizer.zero_grad()

        # Forward pass
        output = model(input_tensor, 0)  # Assuming start_pos=0

        logger.warning(f"Output shape: {output.shape}, output: {output}")

        # Calculate the loss
        loss_fn = torch.nn.CrossEntropyLoss()
        loss = loss_fn(output, target_tensor)

        # Backward pass
        loss.backward()

        # Update the parameters
        optimizer.step()

        logger.warning("Normal Transformer training completed!")


if __name__ == "__main__":
    torch.manual_seed(42)

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-type",
        type=str,
        default="mp",
        choices=["mp", "normal"],
    )
    parser.add_argument(
        "--model-size",
        type=int,
        default=1,
        choices=[1, 3, 8, 0],
    )

    args = vars(parser.parse_args())
    main(args["model_type"], args["model_size"])
