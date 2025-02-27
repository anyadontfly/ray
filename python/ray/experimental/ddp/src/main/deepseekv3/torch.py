import logging
import time

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

    time_dict = {}

    model = Transformer(args)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001)
    logger.debug(f"Model structure: {model}, criterion: {criterion}, optimizer: {optimizer}.")

    num_epochs = 6
    for epoch in range(num_epochs):
        time_dict[epoch] = {}

        fw_start = time.perf_counter()
        outputs = model.forward(random_input, 0)
        fw_end = time.perf_counter()
        time_dict[epoch]["fw"] = fw_end - fw_start

        loss = criterion(outputs, random_target)
        bw_start = time.perf_counter()
        loss.backward()
        bw_end = time.perf_counter()
        time_dict[epoch]["bw"] = bw_end - bw_start

        logger.debug(f"Gradient of first attention layer: {model.layers[0].attn.wo.weight.grad}, shape: {model.layers[0].attn.wo.weight.shape}")

        update_start = time.perf_counter()
        optimizer.step()
        update_end = time.perf_counter()
        time_dict[epoch]["update"] = update_end - update_start
        optimizer.zero_grad()

        logger.debug(f"Params of first attention layer after step: {model.layers[0].attn.wo.weight}")

        logger.info(f"Epoch {epoch}, loss: {loss.item()}")

        total_time = time_dict[epoch]["fw"] + time_dict[epoch]["bw"] + time_dict[epoch]["update"]
        time_dict[epoch]["fw"] = time_dict[epoch]["fw"] / total_time
        time_dict[epoch]["bw"] = time_dict[epoch]["bw"] / total_time
        time_dict[epoch]["update"] = time_dict[epoch]["update"] / total_time
        time_dict[epoch]["total"] = total_time

    for epoch in time_dict.keys():
        logger.warning(time_dict[epoch])


if __name__ == "__main__":
    torch.manual_seed(42)
    main()
