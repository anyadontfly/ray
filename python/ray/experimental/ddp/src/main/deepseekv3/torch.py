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
    # torch.set_default_dtype(torch.bfloat16)
    # torch.set_default_device("cuda")
    # torch.manual_seed(42)
    # args = SMALL
    # x = torch.randint(0, args.vocab_size, (2, 128))
    # model = Transformer(args)
    # output = model(x)
    # print(output, output.size(), output.dtype)

    torch.set_default_dtype(torch.bfloat16)
    # torch.set_default_device("cuda")
    torch.manual_seed(42)
    args = SMALL
    batch_size = 2
    seq_len = 100
    x = torch.randint(0, args.vocab_size, (batch_size, seq_len)).to("cuda")
    model = Transformer(args).to("cuda:0")
    output = model(x)
    print(output, output.size(), output.dtype)

    # criterion = nn.CrossEntropyLoss()
    # optimizer = optim.SGD(model.parameters(), lr=0.001)
    # logger.debug(f"Model structure: {model}, criterion: {criterion}, optimizer: {optimizer}.")

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
