import torch

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
    # Convert weights to half precision
    weights = m.weight.t().half()
    x = x.half()  # Convert input to half precision
    
    x_col_size = x.size()[2]
    weights_row_size = weights.size()[0]
    x_sliced = x[:,:,:x_col_size//split_size]
    weights_sliced = weights[:weights_row_size//split_size,:]
    out = torch.matmul(x_sliced, weights_sliced)

    out = out.squeeze(0)

    for split_idx in range(split_size-1):
        x_sliced = x[:,:,x_col_size*(split_idx+1)//split_size:x_col_size*(split_idx+2)//split_size]
        x_sliced = x_sliced.squeeze(0)
        weights_sliced = weights[weights_row_size*(split_idx+1)//split_size:weights_row_size*(split_idx+2)//split_size,:]
        out.addmm_(x_sliced, weights_sliced)

    return out  # Added return statement

# Update test inputs to use half precision
x = torch.randn([1, 2048, 3584]).half().cuda()  # Changed to half()
m = torch.nn.Linear(in_features=3584, out_features=18944, bias=False).cuda()
m.weight.data = m.weight.data.half()  # Convert model weights to half precision

warmup_size = 20
profile_size = 100

record = Event_record()
for _ in range(warmup_size):
    out = m(x)
with record:
    for _ in range(profile_size):
        out = m(x)
print(f"Time taken: {record.get_time()/profile_size} ms")

split_sizes = [1,2,4,8,16,32,64,128,256,512]
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
