from typing import Any, Dict, List, Union, Tuple, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    import torch

import ray
from ray.dag import (
    DAGNode,
    ClassMethodNode,
)
from ray.dag.constants import COLLECTIVE_OPERATION_KEY
from ray.dag.nccl_operation import _NcclOperation
from ray.experimental.channel import ChannelContext
from ray.experimental.channel.torch_tensor_nccl_channel import _init_communicator
from ray.experimental.channel.torch_tensor_type import Communicator, TorchTensorType
from ray.experimental.util.types import _CollectiveOp, ReduceOp
from ray.util.annotations import DeveloperAPI


class _CollectiveOperation(_NcclOperation):
    """
    Represent metadata for a NCCL collective operation.

    Args:
        input_nodes: A list of input nodes to the collective operation.
        op: The collective operation to perform.
        transport: The transport to use for the collective operation.

    Requirements:
    1. Input nodes are unique.
    2. Actor handles are unique.
    3. Actor handles match the custom NCCL group if specified.
    """

    def __init__(
        self,
        input_nodes: Union[List[DAGNode], List[List[DAGNode]]],
        op: _CollectiveOp,
        transport: Optional[Union[str, Communicator]] = None,
    ):
        super().__init__()

        if len(input_nodes) == 0:
            raise ValueError("Expected input nodes for a collective operation")
        
        if isinstance(input_nodes[0], list):
            assert all(isinstance(input_node, list) for input_node in input_nodes)
            if len(set(len(input_node) for input_node in input_nodes)) != 1:
                raise ValueError(
                    "Expected same number of nodes bound from all actors to be of the same length"
                )
            # TODO: check unique input nodes for each actor
            self.nodes_per_actor = len(input_nodes[0])

        self._actor_handles: List["ray.actor.ActorHandle"] = []
        for input_node in input_nodes:
            if isinstance(input_node, list):
                actor_handle = input_node[0]._get_actor_handle()
            else:
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
        if not isinstance(self._op, ReduceOp):
            raise NotImplementedError("Only ReduceOp is implemented")
        if transport is None:
            transport = TorchTensorType.NCCL
        self._type_hint = TorchTensorType(transport=transport, _direct_return=True)
        if isinstance(transport, Communicator):
            if set(transport.get_actor_handles()) != set(self._actor_handles):
                raise ValueError(
                    "Expected actor handles to match the custom NCCL group"
                )

    def __str__(self) -> str:
        return (
            f"CollectiveGroup("
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

    @property
    def nccl_op_type(self) -> _CollectiveOp:
        return self._op

    def init_communicator(
        self,
        communicator_id: Optional[str] = None,
        use_communication_streams: bool = False,
    ) -> str:
        """
        Initialize the communicator if it has not been initialized yet. If
        `communicator_id` is provided, it means the communicator has already
        been initialized.

        Args:
            communicator_id: The communicator ID, if already initialized.
            use_communication_streams: Whether to use a dedicated stream for
                collective communication. If True, communication and computation
                can be overlapped to improve performance.

        Returns:
            The NCCL group ID.
        """
        type_hint = self._type_hint
        if type_hint.communicator_id is not None:
            return type_hint.communicator_id
        if communicator_id is None:
            communicator_id = _init_communicator(
                self._actor_handles,
                type_hint.get_custom_communicator(),
                use_communication_streams,
            )
        type_hint.set_communicator_id(communicator_id)
        return communicator_id

    def get_communicator(self) -> Communicator:
        if self._type_hint.communicator_id is not None:
            ctx = ChannelContext.get_current()
            communicator = ctx.communicators[self._type_hint.communicator_id]
        elif self._type_hint.get_custom_communicator() is not None:
            communicator = self._type_hint.get_custom_communicator()
        else:
            raise ValueError("Expected a NCCL group")
        return communicator

    def execute(
        self, *send_buf
    ) -> Union["torch.Tensor", Tuple["torch.Tensor", ...]]:
        """
        Call the collective operation on the input tensor(s). Output tensor(s) is
        allocated and returned.
        """
        import torch

        if not (isinstance(send_buf, torch.Tensor) or (isinstance(send_buf, tuple) and all(isinstance(t, torch.Tensor) for t in send_buf))):
            # raise ValueError("Expected a torch tensor")
            # TODO: better error message
            raise ValueError(type(send_buf))

        communicator = self.get_communicator()

        if isinstance(send_buf, torch.Tensor):
            recv_buf = torch.empty_like(send_buf)
            communicator.allreduce(send_buf, recv_buf, self._op)
        else:
            if len(set(t.device for t in send_buf)) != 1:
                raise ValueError("Expected tensors on same device")

            if len(set((t.dtype, t.device) for t in send_buf)) != 1:
                raise ValueError("Expected tensors to have same dtype")
            
            recv_buf = tuple(torch.empty_like(t) for t in send_buf)
            
            coll_stream = torch.cuda.ExternalStream(communicator._coll_stream.ptr)
            copy_stream = torch.cuda.ExternalStream(communicator._copy_stream.ptr)
            copy_to_flatbuf_event = torch.cuda.Event()

            with torch.cuda.stream(copy_stream):
                flat_buf = torch.nn.utils.parameters_to_vector(send_buf)
                copy_to_flatbuf_event.record(copy_stream)

            with torch.cuda.stream(coll_stream):
                coll_stream.wait_event(copy_to_flatbuf_event)

            allreduce_event = communicator.allreduce(flat_buf, flat_buf, self._op, get_event=True)

            with torch.cuda.stream(copy_stream):
                if allreduce_event is not None:
                    copy_stream.wait_event(allreduce_event)
                torch.nn.utils.vector_to_parameters(flat_buf, recv_buf)

        return recv_buf


@DeveloperAPI
class CollectiveOutputNode(ClassMethodNode):
    """Represent an output node from a NCCL collective operation in a Ray DAG."""

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
        super().__init__(
            method_name,
            method_args,
            method_kwargs,
            method_options,
            other_args_to_resolve,
        )

        # Parse the input node.
        # if not (
        #     isinstance(method_args, tuple)
        #     and len(method_args) == 1
        #     and isinstance(method_args[0], DAGNode)
        # ):
        #     raise ValueError("Expected a single input node")
        self._input_node = method_args
        # Parse the collective operation.
        self._collective_op: _CollectiveOperation = other_args_to_resolve.get(
            COLLECTIVE_OPERATION_KEY, None
        )
        if self._collective_op is None:
            raise ValueError("Expected a collective operation")

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
    def nccl_op_type(self) -> _CollectiveOp:
        return self._collective_op.nccl_op_type

    @property
    def nccl_op(self) -> _CollectiveOperation:
        return self._collective_op
