

import llama
import torch
import pandas as pd
from torch.utils.data import Dataset, random_split
from transformers import TrainingArguments, Trainer, AutoTokenizer, AutoModelForCausalLM, AutoConfig
MODEL = "Qwen/Qwen2.5-7B"
DATA_FILE_PATH = 'training_data.csv'

texts = pd.read_csv(DATA_FILE_PATH)['text']

tokenizer = AutoTokenizer.from_pretrained(MODEL)
config = AutoConfig.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_config(config).cuda()
# Add profiler hooks to first decoder layer
def start_profile_hook(module, input):
    torch.cuda.profiler.cudart().cudaProfilerStart()

def stop_profile_hook(module, input, output):
    torch.cuda.profiler.cudart().cudaProfilerStop()

# Get first decoder layer and register hooks
first_decoder = None
for module in model.modules():
    if "Qwen2DecoderLayer" in str(type(module)):
        first_decoder = module
        break

if first_decoder is not None:
    first_decoder.register_forward_pre_hook(start_profile_hook)
    first_decoder.register_forward_hook(stop_profile_hook)


class TextDataset(Dataset):
    def __init__(self, txt_list, tokenizer, max_length):
        self.labels = []
        self.input_ids = []
        self.attn_masks = []
        for txt in txt_list:
            encodings_dict = tokenizer(txt, truncation = True, max_length = max_length, padding = "max_length")
            self.input_ids.append(torch.tensor(encodings_dict['input_ids']))
            self.attn_masks.append(torch.tensor(encodings_dict['attention_mask']))
    def __len__(self): return len(self.input_ids)
    def __getitem__(self, idx): return self.input_ids[idx], self.attn_masks[idx]

dataset = TextDataset(texts, tokenizer, max_length = 2048)

train_dataset, val_dataset = random_split(dataset, [int(0.9 * len(dataset)), len(dataset) - int(0.9 * len(dataset))])

training_args = TrainingArguments(
                                  max_steps = 1,
                                  save_steps = 5000,
                                  warmup_steps = 0,
                                  logging_steps = 100,
                                  weight_decay = 0.05,
                                  num_train_epochs = 1,
                                  logging_dir = './logs',
                                  output_dir = './results',
                                  per_device_eval_batch_size = 1,
                                  per_device_train_batch_size = 1)

Trainer(model = model,
        args = training_args,
        eval_dataset = val_dataset,
        train_dataset = train_dataset,
        data_collator = lambda data: {'input_ids': torch.stack([f[0] for f in data]), 'attention_mask': torch.stack([f[1] for f in data]), 'labels': torch.stack([f[0] for f in data])}).train()

# sample_outputs = model.generate(tokenizer('', return_tensors="pt").input_ids.cuda(),
#                                 do_sample = True,
#                                 top_k = 50,
#                                 max_length = 512,
#                                 top_p = 0.95,
#                                 temperature = 1.0)