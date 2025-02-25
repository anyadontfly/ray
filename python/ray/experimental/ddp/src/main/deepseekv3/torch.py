import logging

import torch
import torch.nn as nn
import torch.optim as optim

from ...core.deepseekv3.model import (
    Transformer,
    SMALL,
)

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="[%(levelname)s %(filename)s:%(lineno)d %(funcName)s] %(message)s",
)

def main():
    torch.set_default_dtype(torch.bfloat16)
    torch.set_default_device("cuda:0")
    torch.manual_seed(42)
    args = SMALL

    batch_size = 2
    seq_len = 100
    random_input = torch.randint(0, args.vocab_size, (batch_size, seq_len))
    random_target = torch.randn(batch_size, seq_len, args.vocab_size)

    model = Transformer(args)
    # output = model(random_input)
    # print(output, output.size(), output.dtype)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001)
    logger.debug(f"Model structure: {model}, criterion: {criterion}, optimizer: {optimizer}.")

    num_epochs = 6
    for epoch in range(num_epochs):
        outputs = model.forward(random_input, 0)

        loss = criterion(outputs, random_target)
        loss.backward()

        logger.debug(f"Gradient of first attention layer: {model.layers[0].attn.wo.weight.grad}, shape: {model.layers[0].attn.wo.weight.shape}")

        optimizer.step()
        optimizer.zero_grad()

        logger.debug(f"Params of first attention layer after step: {model.layers[0].attn.wo.weight}")

        logger.info(f"Epoch {epoch}, loss: {loss.item()}")


if __name__ == "__main__":
    torch.manual_seed(42)
    main()
