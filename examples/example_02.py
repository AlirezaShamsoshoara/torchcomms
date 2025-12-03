# Asynchronous Operations

import torch
from torchcomms import new_comm, ReduceOp

device = torch.device("cuda")
torchcomm = new_comm("nccl", device, name="main_comm")

rank = torchcomm.get_rank()
device_id = rank % torch.cuda.device_count()
target_device = torch.device(f"cuda:{device_id}")

# Create tensor
tensor = torch.full((1024,), float(rank + 1), dtype=torch.float32, device=target_device)

# Start async AllReduce
work = torchcomm.all_reduce(tensor, ReduceOp.SUM, async_op=True)

# Do other work while communication happens
print(f"Rank {rank}: Doing other work while AllReduce is in progress...")

# Wait for completion
work.wait()
print(f"Rank {rank}: AllReduce completed")

torchcomm.finalize()
