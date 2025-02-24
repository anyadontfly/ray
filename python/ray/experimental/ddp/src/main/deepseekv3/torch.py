import logging

import torch
import torch.nn as nn
import torch.optim as optim

from ...core.deepseekv3.model import (
    Transformer,
    ModelArgs,
)

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="[%(levelname)s %(filename)s:%(lineno)d %(funcName)s] %(message)s",
)

def main():
    model_args = ModelArgs(
        dim=512, n_layers=4, n_heads=4, vocab_size=100, max_seq_len=128
    )
    model = Transformer(model_args)

    # Generate random input tensor (batch_size=2, seq_len=128)
    input_tensor = torch.randint(0, model_args.vocab_size, (2, model_args.max_seq_len))

    # Forward pass
    output = model(input_tensor, start_pos=0)

    # Print output shape
    print("Output shape:", output.shape)


    # tokenizer = AutoTokenizer.from_pretrained("./")

    # device = "cuda:0"

    # model_args = SMALL
    # model = Transformer(model_args).to(device)

    # vocab_size = model_args.vocab_size
    # seq_length = model_args.max_seq_len
    # batch_size = model_args.max_batch_size

    # random_input = torch.randint(0, vocab_size, (batch_size, seq_length)).to(device)
    # # random_target = torch.randn(batch_size, seq_length, vocab_size).to(device)
    # outputs = model.forward(random_input, 0)
    # print(outputs.shape)

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
