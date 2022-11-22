#
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#
import os
from typing import Callable, Dict, List, TypeVar
from fairdiplomacy.typedefs import Phase
import tempfile
import submitit


RT = TypeVar("RT")


def submitit_launch_kwargs_sweep(
    F: Callable[..., RT],
    sweep_kwargs: List[Dict],
    slurm_job_name="submitit",
    slurm_partition="Diplomacy",
    **common_kwargs,
) -> List[RT]:
    """Apply F to a list of kwargs using submitit, and return the results.

    Arguments:
        - F: the function to call
        - sweep_kwargs: A list of kwargs dicts to be used in each of the invocations.
        - slurm_job_name: job name for running on slurm.
        - slurm_partition: Partition to run on slurm. If "local", then run locally on this machine.
        - common_kwargs: Any additional kwargs will be passed through to F

    > def add(a, b):
    ...    return a + b
    > submitit_launch_kwargs_sweep(F, sweep_kwargs=[{'a':1},{'a':2}], b=3)
    [4, 5]
    """

    if slurm_partition == "local":
        print("submitit_launch_across_phases: running locally...")
        return [F(**common_kwargs, **job_kwargs) for job_kwargs in sweep_kwargs]

    out_dir = tempfile.mkdtemp(prefix=f"{os.getcwd()}/")
    print(f"submitit_launch_across_phases: temporary files in {out_dir}")

    executor = submitit.AutoExecutor(folder=out_dir)
    executor.update_parameters(
        slurm_job_name=slurm_job_name,
        partition=slurm_partition,
        gpus_per_node=1,
        cpus_per_task=10,
        time=12 * 60,
        constraint="volta32gb",
        # timeout_min=60,
        # slurm_array_parallelism=256,
    )
    with executor.batch():
        jobs = [executor.submit(F, **common_kwargs, **job_kwargs) for job_kwargs in sweep_kwargs]

    print("Waiting on outputs...")
    outputs = [job.result() for job in jobs]
    print("Done!")
    return outputs
