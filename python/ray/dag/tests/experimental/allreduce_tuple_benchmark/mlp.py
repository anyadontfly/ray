import time

import ray
from ray.dag import InputNode, MultiOutputNode
from ray.experimental.collective import allreduce
from ray.air._internal import torch_utils

import torch
import torch.nn as nn


ray.init(num_gpus=2)

class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size, bias=False)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size, bias=False)

    def forward(self, x):
        self.activation = self.relu(self.fc1(x))
        prediction = self.fc2(self.activation)
        return prediction


@ray.remote
class DDPActor:
    def __init__(self, input_size=10, hidden_size=10, output_size=10, learning_rate=0.01):
        self.device = torch_utils.get_devices()[0]
        self.model = MLP(input_size, hidden_size, output_size).to(self.device)
        self.lr = learning_rate
        self.loss_fn = nn.MSELoss()
        self.time = {}
        self.allreduce_size = []

    def forward(self, x):
        self.model.zero_grad()
        time_start = time.perf_counter()
        res = self.model.forward(x)
        time_end = time.perf_counter()
        self.time["forward"] = round(time_end - time_start, 4)
        return res

    def loss(self, prediction, y):
        return self.loss_fn(prediction, y)
    
    @ray.method(num_returns=2)
    def backward_fc2(self, loss):
        time_start = time.perf_counter()
        loss.backward(
            retain_graph=True,
            inputs=[self.model.activation, self.model.fc2.weight]
        )
        time_end = time.perf_counter()
        self.time["backward_fc2"] = round(time_end - time_start, 4)
        self.time["allreduce_fc2"] = time.perf_counter()
        self.allreduce_size.append(round(self.model.fc2.weight.grad.numel() * 4 / (1024 * 1024), 2))
        return self.model.activation.grad, self.model.fc2.weight.grad

    def backward_fc1(self, grad):
        self.time["allreduce_fc2"] = round(time.perf_counter() - self.time["allreduce_fc2"], 4)
        time_start = time.perf_counter()
        self.model.activation.backward(
            gradient=grad,
            inputs=[self.model.fc1.weight],
        )
        time_end = time.perf_counter()
        self.time["backward_fc1"] = round(time_end - time_start, 4)
        self.time["allreduce_fc1"] = time.perf_counter()
        self.allreduce_size.append(round(self.model.fc1.weight.grad.numel() * 4 / (1024 * 1024), 2))
        return self.model.fc1.weight.grad

    def update_fc2(self, grad):
        self.time["allreduce_fc1"] = round(time.perf_counter() - self.time["allreduce_fc1"], 4)
        with torch.no_grad():
            time_start = time.perf_counter()
            self.model.fc2.weight -= self.lr * grad
            time_end = time.perf_counter()
            self.time["update_fc2"] = round(time_end - time_start, 4)

    def update_fc1(self, grad):
        with torch.no_grad():
            time_start = time.perf_counter()
            self.model.fc1.weight -= self.lr * grad
            time_end = time.perf_counter()
            self.time["update_fc1"] = round(time_end - time_start, 4)

    def backward_bucket(self, loss):
        time_start = time.perf_counter()
        loss.backward(
            inputs=[self.model.fc1.weight, self.model.fc2.weight]
        )
        time_end = time.perf_counter()
        self.time["backward_bucket"] = round(time_end - time_start, 4)
        grads = (
            self.model.fc1.weight.grad,
            self.model.fc2.weight.grad,
        )
        self.time["allreduce"] = time.perf_counter()
        self.allreduce_size.append(round((grads[0].numel() + grads[1].numel()) * 4 / (1024 * 1024), 2))
        return grads

    def update_bucket(self, grad_dict):
        self.time["allreduce"] = round(time.perf_counter() - self.time["allreduce"], 4)
        fc1_grad, fc2_grad = grad_dict[0], grad_dict[1]
        with torch.no_grad():
            time_start = time.perf_counter()
            self.model.fc2.weight -= self.lr * fc2_grad
            self.model.fc1.weight -= self.lr * fc1_grad
            time_end = time.perf_counter()
            self.time["update_bucket"] = round(time_end - time_start, 4)

    def stats(self):
        res_time = {}
        res_allreduce_size = self.allreduce_size
        sum = 0
        for _, value in self.time.items():
            sum += value
        for key, value in self.time.items():
            res_time[key] = (value, round(value/sum, 4))
        res_time["total"] = (round(sum, 4), 1)
        self.time = {}
        self.allreduce_size = []
        return res_time, res_allreduce_size


assert (
    sum(node["Resources"].get("GPU", 0) for node in ray.nodes()) > 1
), "This test requires at least 2 GPUs"

torch.manual_seed(42)

input_size = 4096*2
hidden_size = 4096*2
output_size = 4096*2

def load_x1_x2_y1_y2():
    x1 = torch.randn(1, input_size).cuda()
    x2 = torch.randn(1, input_size).cuda()
    y1 = torch.randn(1, output_size).cuda()
    y2 = torch.randn(1, output_size).cuda()
    return x1, x2, y1, y2

actor1 = DDPActor.options(num_gpus=1).remote(input_size, hidden_size, output_size)
actor2 = DDPActor.options(num_gpus=1).remote(input_size, hidden_size, output_size)

with InputNode() as inp:
    x1, x2, y1, y2 = inp.x1, inp.x2, inp.y1, inp.y2
    # forward
    prediction1 = actor1.forward.bind(x1)
    prediction2 = actor2.forward.bind(x2)
    # loss
    loss1 = actor1.loss.bind(prediction1, y1)
    loss2 = actor2.loss.bind(prediction2, y2)
    # backward layer 2
    activation_grad1, fc2_grad1 = actor1.backward_fc2.bind(loss1)
    activation_grad2, fc2_grad2 = actor2.backward_fc2.bind(loss2)
    # allreduce layer 2 gradients
    fc2_grad1, fc2_grad2 = allreduce.bind([fc2_grad1, fc2_grad2])
    # backward layer 1
    fc1_grad1 = actor1.backward_fc1.bind(activation_grad1)
    fc1_grad2 = actor2.backward_fc1.bind(activation_grad2)
    # allreduce layer 1 gradients
    fc1_grad1, fc1_grad2 = allreduce.bind([fc1_grad1, fc1_grad2])
    # update layer 2
    fc2_update1 = actor1.update_fc2.bind(fc2_grad1)
    fc2_update2 = actor2.update_fc2.bind(fc2_grad2)
    # update layer 1
    fc1_update1 = actor1.update_fc1.bind(fc1_grad1)
    fc1_update2 = actor2.update_fc1.bind(fc1_grad2)
    dag_no_bucket = MultiOutputNode([fc1_update1, fc1_update2, fc2_update1, fc2_update2])

with InputNode() as inp:
    x1, x2, y1, y2 = inp.x1, inp.x2, inp.y1, inp.y2
    # forward
    prediction1 = actor1.forward.bind(x1)
    prediction2 = actor2.forward.bind(x2)
    # loss
    loss1 = actor1.loss.bind(prediction1, y1)
    loss2 = actor2.loss.bind(prediction2, y2)
    # backward
    grad_dict1 = actor1.backward_bucket.bind(loss1)
    grad_dict2 = actor2.backward_bucket.bind(loss2)
    # allreduce gradients for both layers
    grad_dict1, grad_dict2 = allreduce.bind([grad_dict1, grad_dict2])
    # update
    update1 = actor1.update_bucket.bind(grad_dict1)
    update2 = actor2.update_bucket.bind(grad_dict2)
    dag_bucket = MultiOutputNode([update1, update2])

compiled_dag = dag_bucket.experimental_compile()
x1, x2, y1, y2 = load_x1_x2_y1_y2()
ref = compiled_dag.execute(x1=x1, x2=x2, y1=y1, y2=y2)
ray.get(ref)

time_total = 0
for i in range(5):
    x1, x2, y1, y2 = load_x1_x2_y1_y2()
    time_start = time.perf_counter()
    ref = compiled_dag.execute(x1=x1, x2=x2, y1=y1, y2=y2)
    ray.get(ref)
    time_end = time.perf_counter()
    time_total += time_end - time_start
    print(f"iter {i} actor breakdown:\n{ray.get(actor1.stats.remote())},\n{ray.get(actor2.stats.remote())}\n")
print(f"Avg time: {time_total/5:.4f} s")
