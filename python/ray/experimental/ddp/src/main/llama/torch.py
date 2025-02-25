import logging

import torch
import torch.nn as nn
import torch.optim as optim

from ...core.llama.model import (
    Transformer,
    SMALL,
)

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="[%(levelname)s %(filename)s:%(lineno)d %(funcName)s] %(message)s",
)

def main():
    device = "cuda:0"

    model_args = SMALL
    model = Transformer(model_args).to(device)

    vocab_size = model_args.vocab_size
    seq_length = model_args.max_seq_len
    batch_size = 32

    random_input = torch.randint(0, vocab_size, (batch_size, seq_length)).to(device)
    random_target = torch.randn(batch_size, seq_length, vocab_size).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001)
    logger.debug(f"Model structure: {model}, criterion: {criterion}, optimizer: {optimizer}.")
    outputs = model.forward(random_input, 0)

    # num_epochs = 6
    # for epoch in range(num_epochs):
    #     outputs = model.forward(random_input, 0)

    #     loss = criterion(outputs, random_target)
    #     loss.backward()

    #     logger.debug(f"Gradient of first attention layer: {model.layers[0].attention.wq.weight.grad}, shape: {model.layers[0].attention.wq.weight.shape}")

    #     optimizer.step()
    #     optimizer.zero_grad()

    #     logger.debug(f"Params of first attention layer after step: {model.layers[0].attention.wq.weight}")

    #     logger.info(f"Epoch {epoch}, loss: {loss.item()}")


if __name__ == "__main__":
    torch.manual_seed(42)

    main()

