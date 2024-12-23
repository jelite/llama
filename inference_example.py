import llama
import torch
import pandas as pd
import os
from torch.utils.data import Dataset, random_split
from transformers import TrainingArguments, Trainer, AutoTokenizer, AutoModelForCausalLM, AutoConfig

MODEL = "Qwen/Qwen2.5-7B"

# Set environment variables for single GPU usage
os.environ["CUDA_VISIBLE_DEVICES"] = "3"

# Initialize tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(MODEL)
config = AutoConfig.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_config(config).half().cuda()

# Add profiler hooks to first decoder layer
def start_profile_hook(module, input):
    import pdb; pdb.set_trace()

    print(f"input size: {input.size()}")
    # torch.cuda.profiler.cudart().cudaProfilerStart()

def stop_profile_hook(module, input, output):
    print(f"input size: {output.size()}")
    # torch.cuda.profiler.cudart().cudaProfilerStop()
torch.Size([1, 2048, 3584])
# Get first decoder layer's gate_proj and register hooks
gate_proj = None
for name, module in model.named_modules():
    if "gate_proj" in name:
        print(name)
        gate_proj = module
        break

if gate_proj is not None:
    gate_proj.register_forward_pre_hook(start_profile_hook)
    gate_proj.register_forward_hook(stop_profile_hook)

# Generate text
batch = tokenizer(" ".join(["Yo"]*2048), return_tensors="pt")
input_ids = batch["input_ids"].cuda()

with torch.cuda.amp.autocast():
    output = model.generate(input_ids, max_new_tokens=2)
print(tokenizer.decode(output[0]))

# Expected output: "Yo mama is so fat, she has to buy two seats on the plane"