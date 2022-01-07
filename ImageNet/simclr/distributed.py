# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

"""Distributed helpers."""
import os
import time
import random
import functools
import logging
import pickle
import socket, fcntl, struct
import torch
import torch.distributed as dist
import warnings

import diffdist.functional as distops

def set_environment_variables_for_nccl_backend(single_node=False, master_port=6105, use_distributed=True):
    if use_distributed:
        assert 'OMPI_COMM_WORLD_RANK' in os.environ
        os.environ['RANK'] = os.environ['OMPI_COMM_WORLD_RANK']
        os.environ['WORLD_SIZE'] = os.environ['OMPI_COMM_WORLD_SIZE']
        os.environ['NCCL_DEBUG'] = 'INFO'
        #os.environ['NCCL_DEBUG_SUBSYS'] = 'COLL'

        for item, value in os.environ.items():
            print('> > {}: {}'.format(item, value))

        #is_philly = 'PHILLY_HOME' in os.environ
        #if is_philly:
        #    set_environment_variables_philly(single_node)

        #is_aml = 'AZ_BATCH_MASTER_NODE' in os.environ
        #if is_aml:
        #    set_environment_variables_aml(single_node, master_port)

        #is_itp = 'DLTS_NUM_WORKER' in os.environ
        #if is_itp:
        #    set_environment_variables_itp(single_node)
    #else:
    #    torch.multiprocessing.set_start_method("forkserver")


def set_environment_variables_itp(single_node=False):
    os.environ['MASTER_ADDR'] = os.environ['MASTER_IP']
    os.environ['MASTER_PORT'] = '54965'


def set_environment_variables_philly(single_node=False):
    # AML-Philly workaround
    os.environ['PHILLY_USE_INFINIBAND'] = 'True'
    os.environ['NCCL_IB_DISABLE'] = '0'
    #os.environ['NCCL_SOCKET_IFNAME'] = os.environ['PHILLY_CONTAINER_ETH_INTERFACES']
    #os.environ['NCCL_IB_HCA'] = os.environ['PHILLY_CONTAINER_IB_HCA']

    IP_INTERFACE_NAME = os.environ['PHILLY_CONTAINER_ETH_INTERFACES']
    print(">>> Rank: {}, IP: {}:{}".format(get_rank(),
        os.environ['PHILLY_CONTAINER_ETH_INTERFACES'],
        get_ip_address(IP_INTERFACE_NAME)))

    philly_job_dir = os.environ['PHILLY_SCRATCH_DIRECTORY']
    nccl_filename = os.path.join(philly_job_dir, 'nccl.info')

    if is_master_proc() and not os.path.isfile(nccl_filename):
        # Get a new MASTER IP address and port for NCCL
        master_ip = get_ip_address(IP_INTERFACE_NAME)
        master_port = random.randint(
                int(os.environ['PHILLY_CONTAINER_PORT_RANGE_START']),
                int(os.environ['PHILLY_CONTAINER_PORT_RANGE_END']))

        with open(nccl_filename, 'w') as fid:
            fid.write('{}:{}'.format(master_ip, master_port))
        print('>>> MASTER: wrote file to {}'.format(nccl_filename))
    else:
        ready = False
        while not ready:
            if os.path.isfile(nccl_filename):
                nccl = open(nccl_filename, 'r').readline().strip().split(':')
                if len(nccl) == 2:
                    ready = True
            time.sleep(2.0)
            print('>>> CLIENT: waiting for {}'.format(nccl_filename))

        master_ip, master_port = nccl[0], nccl[1]

    os.environ['MASTER_ADDR'] = str(master_ip)
    os.environ['MASTER_PORT'] = str(master_port)
    print('>>> [rank: {}] MASTER NODE: {}:{}'.format(get_rank(), master_ip, master_port))


def set_environment_variables_aml(single_node=False, master_port=6105):
    if not single_node:
        master_node_params = os.environ['AZ_BATCH_MASTER_NODE'].split(':')
        os.environ['MASTER_ADDR'] = master_node_params[0]
        # Do not overwrite master port with that defined in AZ_BATCH_MASTER_NODE
        if 'MASTER_PORT' not in os.environ:
            os.environ['MASTER_PORT'] = str(master_port)
    else:
        os.environ['MASTER_ADDR'] = os.environ['AZ_BATCHAI_MPI_MASTER_NODE']
        os.environ['MASTER_PORT'] = '54965'
    print('NCCL_SOCKET_IFNAME original value = {}'.format(os.environ['NCCL_SOCKET_IFNAME']))
    # TODO make this parameterizable
    os.environ['NCCL_SOCKET_IFNAME'] = '^docker0,lo'

    print('RANK = {}'.format(os.environ['RANK']))
    print('WORLD_SIZE = {}'.format(os.environ['WORLD_SIZE']))
    print('MASTER_ADDR = {}'.format(os.environ['MASTER_ADDR']))
    print('MASTER_PORT = {}'.format(os.environ['MASTER_PORT']))
    print('NCCL_SOCKET_IFNAME new value = {}'.format(os.environ['NCCL_SOCKET_IFNAME']))
    # print('MASTER_NODE = {}'.format(os.environ['MASTER_NODE']))


def all_gather(tensors):
    """
    All gathers the provided tensors from all processes across machines.
    Args:
        tensors (list): tensors to perform all gather across all processes in
        all machines.
    """

    gather_list = []
    output_tensor = []
    world_size = dist.get_world_size()
    for tensor in tensors:
        tensor_placeholder = [
            torch.ones_like(tensor) for _ in range(world_size)
        ]
        dist.all_gather(tensor_placeholder, tensor, async_op=False)
        gather_list.append(tensor_placeholder)
    for gathered_tensor in gather_list:
        output_tensor.append(torch.cat(gathered_tensor, dim=0))
    return output_tensor


def all_reduce(tensors, average=True):
    """
    All reduce the provided tensors from all processes across machines.
    Args:
        tensors (list): tensors to perform all reduce across all processes in
        all machines.
        average (bool): scales the reduced tensor by the number of overall
        processes across all machines.
    """

    for tensor in tensors:
        dist.all_reduce(tensor, async_op=False)
    if average:
        world_size = dist.get_world_size()
        for tensor in tensors:
            tensor.mul_(1.0 / world_size)
    return tensors


def init_process_group(
    local_rank,
    local_world_size,
    shard_id,
    num_shards,
    init_method,
    dist_backend="nccl",
):
    """
    Initializes the default process group.
    Args:
        local_rank (int): the rank on the current local machine.
        local_world_size (int): the world size (number of processes running) on
        the current local machine.
        shard_id (int): the shard index (machine rank) of the current machine.
        num_shards (int): number of shards for distributed training.
        init_method (string): supporting three different methods for
            initializing process groups:
            "file": use shared file system to initialize the groups across
            different processes.
            "tcp": use tcp address to initialize the groups across different
        dist_backend (string): backend to use for distributed training. Options
            includes gloo, mpi and nccl, the details can be found here:
            https://pytorch.org/docs/stable/distributed.html
    """
    # Sets the GPU to use.
    torch.cuda.set_device(local_rank)
    # Initialize the process group.
    proc_rank = local_rank + shard_id * local_world_size
    world_size = local_world_size * num_shards
    dist.init_process_group(
        backend=dist_backend,
        init_method=init_method,
        world_size=world_size,
        rank=proc_rank,
    )


def get_ip_address(ifname):
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    return socket.inet_ntoa(fcntl.ioctl(
        s.fileno(),
        0x8915,  # SIOCGIFADDR
        struct.pack('256s', ifname[:15].encode('utf-8'))
    )[20:24])


def is_master_proc():
    """
    Determines if the current process is the master process.
    """
    return get_rank() == 0


def get_world_size():
    """
    Get the size of the world.
    """
    if 'WORLD_SIZE' in os.environ:
        return int(os.environ['WORLD_SIZE'])
    else:
        if not dist.is_available():
            return 1
        if not dist.is_initialized():
            return 1
        return dist.get_world_size()


def get_global_size():
    if 'OMPI_COMM_WORLD_SIZE' in os.environ:
        return int(os.environ['OMPI_COMM_WORLD_SIZE'])
    else:
        return get_world_size()


def get_local_size(num_gpus=None):
    if 'OMPI_COMM_WORLD_LOCAL_SIZE' in os.environ:
        return int(os.environ['OMPI_COMM_WORLD_LOCAL_SIZE'])
    else:
        if num_gpus is None:
            num_gpus = torch.cuda.device_count()
        return get_global_size() // num_gpus


def get_rank():
    """
    Get the rank of the current process.
    """
    if 'OMPI_COMM_WORLD_RANK' in os.environ:
        return int(os.environ['OMPI_COMM_WORLD_RANK'])
    else:
        if not dist.is_available():
            return 0
        if not dist.is_initialized():
            return 0
        return dist.get_rank()


def get_local_rank(num_gpus=None):
    """
    Get the local rank of the current process.
    """
    if 'OMPI_COMM_WORLD_LOCAL_RANK' in os.environ:
        return int(os.environ['OMPI_COMM_WORLD_LOCAL_RANK'])
    else:
        if num_gpus is None:
            num_gpus = torch.cuda.device_count()
        if not dist.is_available():
            return 0
        if not dist.is_initialized():
            return 0
        return dist.get_rank() % num_gpus


def synchronize():
    """
    Helper function to synchronize (barrier) among all processes when
    using distributed training
    """
    if not dist.is_available():
        return
    if not dist.is_initialized():
        return
    world_size = dist.get_world_size()
    if world_size == 1:
        return
    dist.barrier()


@functools.lru_cache()
def _get_global_gloo_group():
    """
    Return a process group based on gloo backend, containing all the ranks
    The result is cached.
    Returns:
        (group): pytorch dist group.
    """
    if dist.get_backend() == "nccl":
        return dist.new_group(backend="gloo")
    else:
        return dist.group.WORLD


def _serialize_to_tensor(data, group):
    """
    Seriialize the tensor to ByteTensor. Note that only `gloo` and `nccl`
        backend is supported.
    Args:
        data (data): data to be serialized.
        group (group): pytorch dist group.
    Returns:
        tensor (ByteTensor): tensor that serialized.
    """

    backend = dist.get_backend(group)
    assert backend in ["gloo", "nccl"]
    device = torch.device("cpu" if backend == "gloo" else "cuda")

    buffer = pickle.dumps(data)
    if len(buffer) > 1024 ** 3:
        logger = logging.getLogger(__name__)
        logger.warning(
            "Rank {} trying to all-gather {:.2f} GB of data on device {}".format(
                get_rank(), len(buffer) / (1024 ** 3), device
            )
        )
    storage = torch.ByteStorage.from_buffer(buffer)
    tensor = torch.ByteTensor(storage).to(device=device)
    return tensor


def _pad_to_largest_tensor(tensor, group):
    """
    Padding all the tensors from different GPUs to the largest ones.
    Args:
        tensor (tensor): tensor to pad.
        group (group): pytorch dist group.
    Returns:
        list[int]: size of the tensor, on each rank
        Tensor: padded tensor that has the max size
    """
    world_size = dist.get_world_size(group=group)
    assert (
        world_size >= 1
    ), "comm.gather/all_gather must be called from ranks within the given group!"
    local_size = torch.tensor(
        [tensor.numel()], dtype=torch.int64, device=tensor.device
    )
    size_list = [
        torch.zeros([1], dtype=torch.int64, device=tensor.device)
        for _ in range(world_size)
    ]
    dist.all_gather(size_list, local_size, group=group)
    size_list = [int(size.item()) for size in size_list]

    max_size = max(size_list)

    # we pad the tensor because torch all_gather does not support
    # gathering tensors of different shapes
    if local_size != max_size:
        padding = torch.zeros(
            (max_size - local_size,), dtype=torch.uint8, device=tensor.device
        )
        tensor = torch.cat((tensor, padding), dim=0)
    return size_list, tensor


def all_gather_unaligned(data, group=None):
    """
    Run all_gather on arbitrary picklable data (not necessarily tensors).

    Args:
        data: any picklable object
        group: a torch process group. By default, will use a group which
            contains all ranks on gloo backend.

    Returns:
        list[data]: list of data gathered from each rank
    """
    if get_world_size() == 1:
        return [data]
    if group is None:
        group = _get_global_gloo_group()
    if dist.get_world_size(group) == 1:
        return [data]

    tensor = _serialize_to_tensor(data, group)

    size_list, tensor = _pad_to_largest_tensor(tensor, group)
    max_size = max(size_list)

    # receiving Tensor from all ranks
    tensor_list = [
        torch.empty((max_size,), dtype=torch.uint8, device=tensor.device)
        for _ in size_list
    ]
    dist.all_gather(tensor_list, tensor, group=group)

    data_list = []
    for size, tensor in zip(size_list, tensor_list):
        buffer = tensor.cpu().numpy().tobytes()[:size]
        data_list.append(pickle.loads(buffer))

    return data_list


def worker_urls(urls):
    """Selects a subset of urls based on Torch get_worker_info.
    Used as a shard selection function in Dataset."""

    assert isinstance(urls, list)
    assert isinstance(urls[0], str)

    worker_info = torch.utils.data.get_worker_info()
    if worker_info is not None:
        wid = worker_info.id
        num_workers = worker_info.num_workers
        if wid == 0 and len(urls) < num_workers:
            warnings.warn(f"num_workers {num_workers} > num_shards {len(urls)}")
        return urls[wid::num_workers]
    else:
        return urls


def shard_selection_per_node(full_urls):
    assert isinstance(full_urls, list)
    assert isinstance(full_urls[0], str)

    index, total = get_rank(), get_world_size()
    wrap_around_list = [x % total for x in range(index, index + total)]
    urls = []
    for i in wrap_around_list:
        urls += full_urls[i::total]

    return urls

def shard_selection(full_urls):
    return worker_urls(shard_selection_per_node(full_urls))


def diff_all_gather(tensor, dim=0):
    tensor_placeholder = [
        torch.zeros_like(tensor) for i in range(get_world_size())
    ]
    tensor_placeholder = distops.all_gather(
        tensor_placeholder, tensor
    )
    return torch.cat(tensor_placeholder, dim=dim)
