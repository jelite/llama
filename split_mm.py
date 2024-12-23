import torch
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--data_type', type=str, default='half', choices=['half', 'bf16', 'full'],
                   help='Data type for computation: half, bf16, or full precision')
parser.add_argument('--is_profile', type=bool, default=False, choices=[True, False],
                   help='Whether to profile the computation')
args = parser.parse_args()

data_type = args.data_type
is_profile = args.is_profile

print(f"data_type: {data_type}, is_profile: {is_profile}")

class Event_record():
    def __init__(self):
        self.start = torch.cuda.Event(enable_timing=True)
        self.end = torch.cuda.Event(enable_timing=True)
    def record_start(self):
        self.start.record()
    def record_end(self):
        self.end.record()
    def __enter__(self):
        self.record_start()
    def __exit__(self, exc_type, exc_value, traceback):
        self.record_end()
        torch.cuda.synchronize()
    def get_time(self):
        return self.start.elapsed_time(self.end)

def splited_linear(x, m, split_size):
    if data_type == "half":
        weights = m.weight.t().half()
        x = x.half()
    elif data_type == "bf16":
        weights = m.weight.t().bfloat16()
        x = x.bfloat16()
    elif data_type == "full":
        weights = m.weight.t()
        x = x.float()
    x_col_size = x.size()[2]
    weights_row_size = weights.size()[0]
    x_sliced = x[:,:,:x_col_size//split_size]
    weights_sliced = weights[:weights_row_size//split_size,:]
    out = torch.matmul(x_sliced, weights_sliced)

    if is_profile != True:
        out = out.squeeze(0)
        for split_idx in range(split_size-1):
            x_sliced = x[:,:,x_col_size*(split_idx+1)//split_size:x_col_size*(split_idx+2)//split_size]
            x_sliced = x_sliced.squeeze(0)
            weights_sliced = weights[weights_row_size*(split_idx+1)//split_size:weights_row_size*(split_idx+2)//split_size,:]
            out.addmm_(x_sliced, weights_sliced)

    # print(f"Are they equal? {torch.allclose(out, torch.matmul(x,weights))}")

    return out

if data_type == "half":
    x = torch.randn([1, 2048, 3584]).half().cuda()
    m = torch.nn.Linear(in_features=3584, out_features=18944, bias=False).cuda()
    m.weight.data = m.weight.data.half()
elif data_type == "bf16":
    x = torch.randn([1, 2048, 3584]).bfloat16().cuda()
    m = torch.nn.Linear(in_features=3584, out_features=18944, bias=False).cuda()
    m.weight.data = m.weight.data.bfloat16()
elif data_type == "full":
    x = torch.randn([1, 2048, 3584]).float().cuda()
    m = torch.nn.Linear(in_features=3584, out_features=18944, bias=False).cuda()

warmup_size = 0
profile_size = 1

record = Event_record()
# for _ in range(warmup_size):
#     out = m(x)
# with record:
#     for _ in range(profile_size):
#         out = m(x)
# print(f"Time taken: {record.get_time()/profile_size} ms")

split_sizes = [1,2,4,8,16,32,64,128,256]
# split_sizes = [32]
for split_size in split_sizes:
    for _ in range(warmup_size):
        out = splited_linear(x, m, split_size)
    with record:
        for _ in range(profile_size):
            out = splited_linear(x, m, split_size)
    print(f"{split_size}, {record.get_time()/profile_size}")




print("##############")
# lin = torch.nn.Linear(2, 7, bias=False)
# lin.weight = torch.nn.Parameter(w.t())
# out = lin(x)
# print(out)


# print(f"x: {x}")
# print(f"y: {y}")
# print(f"z: {z}")
# # torch.addcmul(x,y,z,_,1,x)
# o = torch.mul(y,z) + x
# print(f"o: {o}")
# out = x.addcmul_(y, z)
# print(out)

# print(out.data_ptr() == x.data_ptr()) # prints True
