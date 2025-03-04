import logging
from typing import Any, Dict, List

import torch

import ray
from ....core.common import get_end_time, get_start_time, log_elapses_to_csv
from ....core.config import parse_args
from ....core.linear.actor import LinearActor
from ray.dag import InputNode, MultiOutputNode
from ray.experimental.collective import allgather, reducescatter

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.WARNING,
    format="[%(levelname)s %(filename)s:%(lineno)d %(funcName)s] %(message)s",
)
logger.info("Welcome to Downton Abbey!")


def init_actors(args: Dict[str, Any]) -> List[LinearActor]:
    layer_size = args["layer_size"]
    num_layers = args["num_layers"]
    num_units = args["num_partitions"]
    num_actors = args["num_actors"]
    device = "cuda:0"
    tracing = args["tracing"]

    actor_cls = LinearActor.options(num_gpus=1)
    actors = [
        actor_cls.remote(
            layer_size=layer_size,
            num_layers=num_layers,
            num_units=num_units,
            num_actors=num_actors,
            device=device,
            tracing=tracing,
        )
        for _ in range(num_actors)
    ]

    return actors


def get_metrics(tracing: bool) -> List[str]:
    if not tracing:
        metrics = [
            "total",
            "actor.total",
        ]
    else:
        metrics = [
            "total",
            "actor.total",
            "fw.total",
            "bw.total",
            "bw.loss",
            "bw.grad",
            "bw.others",
            "bw.upd",
        ]
    return metrics


def train(
    actors: List[LinearActor],
    num_units: int,
    num_iters: int,
    output_path: str,
    latency_prefix: str,
    save_model: bool,
    model_prefix: str,
    tracing: bool,
) -> None:
    with InputNode() as inp:
        inputs = [actor.get_input.bind(inp) for actor in actors]
        shards = [actor.get_shard.bind(0, inp) for actor in actors]
        params = allgather.bind(shards)
        for idx in range(num_units):
            if idx < num_units - 1:
                shards_pf = [actor.get_shard.bind(idx + 1, inp) for actor in actors]
                params_pf = allgather.bind(shards_pf)
            inputs = [
                actor.forward.bind(idx, param, input)
                for actor, param, input in zip(actors, params, inputs)
            ]
            if idx < num_units - 1:
                params = params_pf

        targets = [actor.get_target.bind(inp) for actor in actors]
        losses = [
            actor.compute_loss.bind(output, target)
            for actor, output, target in zip(actors, inputs, targets)
        ]

        updates = []
        for idx in reversed(range(num_units)):
            if idx > 0:
                shards_pf = [actor.get_shard.bind(idx - 1, inp) for actor in actors]
                params_pf = allgather.bind(shards_pf)
            if idx == num_units - 1:
                grads = [
                    actor.backward_loss.bind(loss)
                    for actor, loss in zip(actors, losses)
                ]
            else:
                grads = [
                    actor.backward.bind(idx, param)
                    for actor, param in zip(actors, params)
                ]
            reduced_grads = reducescatter.bind(grads)
            updates.extend(
                [
                    actor.update.bind(idx, grad, True)
                    for actor, grad in zip(actors, reduced_grads)
                ]
            )
            if idx > 0:
                params = params_pf

        dag = MultiOutputNode(updates)

    compiled_dag = dag.experimental_compile(_overlap_gpu_communication=True)
    actor_to_shards = ray.get(actors[0].init_and_shard_model.remote())
    for actor, shards in zip(actors, actor_to_shards):
        ray.get(actor.set_shards.remote(shards))

    total_elapses: List[int] = []
    for iter in range(num_iters):
        for actor in actors:
            ray.get(actor.init_training.remote())
            ray.get(actor.init_tracing.remote())

        start = get_start_time()
        compiled_dag.execute(None)
        end = get_end_time()
        elapse_us = round((end - start) * 1e6)

        if save_model:
            for i, actor in enumerate(actors):
                weights = ray.get(actor.fetch_weights.remote())
                for idx, weight in enumerate(weights):
                    logger.info(f"actor: {i}, layer: {idx}, shard: {weight}")

        if iter > 0:
            logger.warning(f"iter: {iter}, elapse: {elapse_us} us")
            total_elapses.append(elapse_us)

        for actor in actors:
            ray.get(actor.finish_tracing.remote())

    actors_to_elapses = [ray.get(actor.fetch_traces.remote()) for actor in actors]
    for actor_elapses in actors_to_elapses:
        actor_elapses["total"] = total_elapses
    log_elapses_to_csv(
        actors_to_elapses,
        output_path,
        latency_prefix,
        get_metrics(tracing),
    )

    if save_model:
        model_file = f"{model_prefix}.log"
        with open(model_file, "w") as f:
            for weight in weights:
                f.write(f"{weight}\n")
        for i, actor in enumerate(actors):
            weights = ray.get(actor.fetch_weights.remote())
            model_file = f"{model_prefix}_{i}.log"
            with open(model_file, "w") as f:
                for weight in weights:
                    f.write(f"{weight}\n")


def main(args: Dict[str, Any]) -> None:
    ray.init()

    actors = init_actors(args)

    train(
        actors,
        args["num_partitions"],
        args["num_iters"],
        args["output_path"],
        args["latency_prefix"],
        args.get("save_model", False),
        args["model_prefix"],
        args["tracing"],
    )

    for actor in actors:
        ray.kill(actor)
    ray.shutdown()


if __name__ == "__main__":
    args = parse_args()
    main(args)
