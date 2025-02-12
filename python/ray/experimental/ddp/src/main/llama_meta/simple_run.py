import argparse
from ...core.llama_meta.model import (
    Transformer,
    TransformerMP,
    LLAMA_1B,
    LLAMA_3B,
    LLAMA_8B
)

from typing import Any, Dict

import torch


def main(args: Dict[str, Any]) -> None:
    if args.model_type == "mp":
        print("Bucket training started!")

        if args.model_size == 1:
            model = TransformerMP(LLAMA_1B).to("cuda")
        elif args.model_size == 3:
            model = TransformerMP(LLAMA_3B).to("cuda")
        else:
            model = TransformerMP(LLAMA_8B).to("cuda")

        # Assuming you have a TransformerMP model and input data
        input_tensor = torch.randint(0, 128256, (1, 2048)).to("cuda")
        target_tensor = torch.randn(1, 2048, 128256, requires_grad=True).to("cuda")

        output = input_tensor

        # # Iterate through the buckets
        # for bucket in model.bucket_params:
        #     # Zero the gradients for the bucket's optimizer
        #     bucket.optimizer.zero_grad()

        #     # Forward pass through the bucket
        #     output = bucket(output)

        output = model(input_tensor, 0)

        print(f"Output shape: {output.shape}, output: {output}")

        # for bucket in reversed(model.bucket_params):


        # # Calculate the loss (example using cross-entropy loss)
        # loss_fn = torch.nn.CrossEntropyLoss()
        # loss = loss_fn(output, target_tensor)

        # # Backward pass through the bucket
        # loss.backward()

        # # Update the parameters in the bucket
        # bucket.optimizer.step()

        print("Bucket training completed!")

    else:
        print("Normal Transformer training started!")

        if args.model_size == 1:
            model = Transformer(LLAMA_1B).to("cuda")
        elif args.model_size == 3:
            model = Transformer(LLAMA_3B).to("cuda")
        else:
            model = Transformer(LLAMA_8B).to("cuda")

        # Assuming you have a Transformer model and input data
        input_tensor = torch.randint(0, 128256, (1, 2048)).to("cuda")
        target_tensor = torch.randn(1, 2048, 128256, requires_grad=True).to("cuda")

        # Define the optimizer
        optimizer = torch.optim.Adam(model.parameters())

        # Zero the gradients
        optimizer.zero_grad()

        # Forward pass
        output = model(input_tensor, 0)  # Assuming start_pos=0

        print(f"Output shape: {output.shape}, output: {output}")

        # Calculate the loss
        loss_fn = torch.nn.CrossEntropyLoss()
        loss = loss_fn(output, target_tensor)

        # Backward pass
        loss.backward()

        # Update the parameters
        optimizer.step()

        print("Normal Transformer training completed!")


if __name__ == "__main__":
    torch.manual_seed(42)

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_type",
        type=str,
        default="mp",
        choices=["mp", "normal"],
    )
    parser.add_argument(
        "--model_size",
        type=int,
        default=1,
        choices=[1, 3, 8],
    )
    args = parser.parse_args()
    main(args)
