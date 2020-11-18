import os
import torch
import argparse
import skip_models
import numpy as np
import torch
import torch.nn as nn
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data.distributed import DistributedSampler
import torch.nn.functional as F
import torch_geometric.transforms as T
import torch_geometric.nn as pyg_nn
import torch.optim as optim


# use for graph or node classification where you want to automatically split batches to each worker
def model_parallelize_with_split(dset, model, rank, num_workers = None, addr = 'localhost', prt = '12355', backend = 'nccl'):
    dset[0] = dset[0].split(dset[0].size(0) // world_size)[rank]
    return model_parallelize(model,rank,num_workers,addr,prt,backend), dset

# use when you want to deal with splitting data among workers yourself, (i.e. for link prediction)
def model_parallelize(model, rank, num_workers = None, addr = 'localhost', prt = '12355', backend = 'nccl', single_node=False):
    if num_workers != None:
        world_size = torch.cuda.device_count()
    else:
        world_size = num_workers
    os.environ['MASTER_ADDR'] = addr
    os.environ['MASTER_PORT'] = prt
    dist.init_process_group(backend, rank=rank, world_size=world_size)
    if single_node:
        model = DistributedDataParallel(model,device_ids=[rank])
    else: 
        model = DistributedDataParallel(model,
                                        device_ids=[args.local_rank],
                                        output_device=args.local_rank)
    return model
    
# use to overwrite default gradient synchronizer with your own reduce operation
def parallel_sync(model, syncop = torch.distributed.ReduceOp.SUM, divide_by_n = True,n_gpus):
    for param in model.parameters():
        if param.requires_grad and param.grad is not None:
            torch.distributed.all_reduce(param.grad.data, op=syncop)
            if divide_by_n:
                param.grad.data /= n_gpus
