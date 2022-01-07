# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

"""Multiprocessing helpers."""

import os
import torch
import distributed as du


def run(
    local_rank, func, backend, model, dm
):
    """
    Runs a function from a child process.
    Args:
        local_rank (int): rank of the current process on the current machine.
        func (function): function to execute on each of the process.
        backend (string): three distributed backends ('nccl', 'gloo', 'mpi') are
            supports, each with different capabilities. Details can be found
            here:
            https://pytorch.org/docs/stable/distributed.html
        cfg (CfgNode): configs. Details can be found in
            slowfast/config/defaults.py
    """
    # master_url = "tcp://" + os.environ['MASTER_ADDR'] + ":" + os.environ['MASTER_PORT']
    try:
        master_url = "tcp://" + os.environ['MASTER_ADDR'] + ":" + os.environ['MASTER_PORT']
        torch.distributed.init_process_group(
            backend=backend,
            init_method=master_url,
            rank=du.get_rank(),
            world_size=du.get_global_size()
        )
        '''
        if 'PHILLY_HOME' in os.environ:
            master_url = "tcp://" + os.environ['MASTER_ADDR'] + ":" + os.environ['MASTER_PORT']
            torch.distributed.init_process_group(
                backend=backend,
                init_method=master_url,
                rank=du.get_rank(),
                world_size=du.get_global_size()
            )
        else:
            torch.distributed.init_process_group(
                backend=backend,
            )
        '''
    except Exception as e:
        raise e

    torch.cuda.set_device(local_rank)
    func(model, dm)


def run_local(
    local_rank, num_proc, func, init_method, shard_id, num_shards, backend, cfg
):
    """
    Runs a function from a child process.
    Args:
        local_rank (int): rank of the current process on the current machine.
        num_proc (int): number of processes per machine.
        func (function): function to execute on each of the process.
        init_method (string): method to initialize the distributed training.
            TCP initialization: equiring a network address reachable from all
            processes followed by the port.
            Shared file-system initialization: makes use of a file system that
            is shared and visible from all machines. The URL should start with
            file:// and contain a path to a non-existent file on a shared file
            system.
        shard_id (int): the rank of the current machine.
        num_shards (int): number of overall machines for the distributed
            training job.
        backend (string): three distributed backends ('nccl', 'gloo', 'mpi') are
            supports, each with different capabilities. Details can be found
            here:
            https://pytorch.org/docs/stable/distributed.html
        cfg (CfgNode): configs. Details can be found in
            slowfast/config/defaults.py
    """
    # Initialize the process group.
    world_size = num_proc * num_shards
    rank = shard_id * num_proc + local_rank

    try:
        torch.distributed.init_process_group(
            backend=backend,
            init_method=init_method,
            world_size=world_size,
            rank=rank,
        )
    except Exception as e:
        raise e

    torch.cuda.set_device(local_rank)
    func(cfg)
