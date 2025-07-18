from typing import Any, Dict, List, Union, Tuple, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    import torch

import ray
from ray.dag import (
    DAGNode,
    ClassMethodNode,
)
from ray.dag.constants import COLLECTIVE_OPERATION_KEY
from ray.experimental.channel import ChannelContext
from ray.experimental.channel.torch_tensor_type import Communicator, TorchTensorType
from ray.experimental.util.types import (
    _CollectiveOp,
    AllGatherOp,
    AllReduceOp,
    ReduceScatterOp,
)
from ray.util.annotations import DeveloperAPI


class _CollectiveOperation:
    """
    Represent metadata for a collective communicator collective operation.

    Args:
        input_nodes: A list of input nodes to the collective operation.
        op: The collective operation to perform.
        transport: The transport to use for the collective operation.

    Requirements:
    1. Input nodes are unique.
    2. Actor handles are unique.
    3. Actor handles match the custom communicator group if specified.
    """

    def __init__(
        self,
        input_nodes: List[DAGNode],
        op: _CollectiveOp,
        transport: Optional[Union[str, Communicator]] = None,
    ):
        if len(input_nodes) == 0:
            raise ValueError("Expected input nodes for a collective operation")
        if len(set(input_nodes)) != len(input_nodes):
            raise ValueError("Expected unique input nodes for a collective operation")

        self._actor_handles: List["ray.actor.ActorHandle"] = []
        for input_node in input_nodes:
            actor_handle = input_node._get_actor_handle()
            if actor_handle is None:
                raise ValueError("Expected an actor handle from the input node")
            self._actor_handles.append(actor_handle)
        if len(set(self._actor_handles)) != len(self._actor_handles):
            invalid_input_nodes = [
                input_node
                for input_node in input_nodes
                if self._actor_handles.count(input_node._get_actor_handle()) > 1
            ]
            raise ValueError(
                "Expected unique actor handles for a collective operation, "
                "but found duplicate actor handles from input nodes: "
                f"{invalid_input_nodes}"
            )

        self._op = op
        if transport is None:
            transport = TorchTensorType.ACCELERATOR
        self._type_hint = TorchTensorType(transport=transport, _direct_return=True)
        if isinstance(transport, Communicator):
            if set(transport.get_actor_handles()) != set(self._actor_handles):
                raise ValueError(
                    "Expected actor handles to match the custom communicator group"
                )

    def __str__(self) -> str:
        return (
            f"CollectiveOperation("
            f"_actor_handles={self._actor_handles}, "
            f"_op={self._op}, "
            f"_type_hint={self._type_hint})"
        )

    @property
    def actor_handles(self) -> List["ray.actor.ActorHandle"]:
        return self._actor_handles

    @property
    def type_hint(self) -> TorchTensorType:
        return self._type_hint

    def get_communicator(self) -> Communicator:
        if self._type_hint.communicator_id is not None:
            ctx = ChannelContext.get_current()
            communicator = ctx.communicators[self._type_hint.communicator_id]
        elif self._type_hint.get_custom_communicator() is not None:
            communicator = self._type_hint.get_custom_communicator()
        else:
            raise ValueError("Expected a communicator group")
        return communicator

    def execute(self, send_buf: "torch.Tensor") -> "torch.Tensor":
        """
        Call the collective operation on the input tensor. An output tensor is
        allocated and returned.
        """
        import torch

        if not isinstance(send_buf, torch.Tensor):
            raise ValueError("Expected a torch tensor")
        communicator = self.get_communicator()

        if isinstance(self._op, AllGatherOp):
            world_size = len(self._actor_handles)
            recv_buf = torch.empty(
                (send_buf.shape[0] * world_size, *send_buf.shape[1:]),
                dtype=send_buf.dtype,
                device=send_buf.device,
            )
            communicator.allgather(send_buf, recv_buf)
        elif isinstance(self._op, AllReduceOp):
            recv_buf = torch.empty_like(send_buf)
            communicator.allreduce(send_buf, recv_buf, self._op.reduceOp)
        elif isinstance(self._op, ReduceScatterOp):
            world_size = len(self._actor_handles)
            if send_buf.shape[0] % world_size != 0:
                raise ValueError(
                    "Expected the first dimension of the input tensor to be divisible "
                    f"by the world size {world_size}"
                )
            recv_buf = torch.empty(
                (send_buf.shape[0] // world_size, *send_buf.shape[1:]),
                dtype=send_buf.dtype,
                device=send_buf.device,
            )
            communicator.reducescatter(send_buf, recv_buf, self._op.reduceOp)
        else:
            raise ValueError("Expected a collective operation")
        return recv_buf


@DeveloperAPI
class CollectiveOutputNode(ClassMethodNode):
    """Represent an output node from a communicator collective operation in a Ray DAG."""

    def __init__(
        self,
        method_name: str,
        method_args: Tuple[
            DAGNode,
        ],
        method_kwargs: Dict[str, Any],
        method_options: Dict[str, Any],
        other_args_to_resolve: Dict[str, Any],
    ):
        # Parse the input node.
        if not (
            isinstance(method_args, tuple)
            and len(method_args) == 1
            and isinstance(method_args[0], DAGNode)
        ):
            raise ValueError("Expected a single input node")
        self._input_node = method_args[0]
        # Parse the collective operation.
        self._collective_op: _CollectiveOperation = other_args_to_resolve.get(
            COLLECTIVE_OPERATION_KEY, None
        )
        if self._collective_op is None:
            raise ValueError("Expected a collective operation")

        super().__init__(
            method_name,
            method_args,
            method_kwargs,
            method_options,
            other_args_to_resolve,
        )

    def _copy_impl(
        self,
        new_args: List[Any],
        new_kwargs: Dict[str, Any],
        new_options: Dict[str, Any],
        new_other_args_to_resolve: Dict[str, Any],
    ):
        return CollectiveOutputNode(
            self._method_name,
            new_args,
            new_kwargs,
            new_options,
            other_args_to_resolve=new_other_args_to_resolve,
        )

    def _execute_impl(self, *args, **kwargs):
        raise NotImplementedError(
            "CollectiveOutputNode is only supported with dag.experimental_compile()"
        )

    @property
    def collective_op(self) -> _CollectiveOperation:
        return self._collective_op
