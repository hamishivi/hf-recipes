import re
import torch
import random
from transformers import GPT2LMHeadModel, AutoTokenizer, DataCollatorForSeq2Seq, OPTForCausalLM
from transformers import Seq2SeqTrainingArguments, Seq2SeqTrainer, logging
from datasets import load_dataset
from ni_collator import DataCollatorForNI
from evaluation import compute_metrics
from transformers import pipeline
import json
from tqdm import tqdm
from transformers.pipelines.pt_utils import KeyDataset


#model = GPT2LMHeadModel.from_pretrained('test/checkpoint-9426')#.to('cuda')
model = OPTForCausalLM.from_pretrained('opt13b/checkpoint-1767', device_map="auto")
tokenizer = AutoTokenizer.from_pretrained('facebook/opt-1.3b', return_tensors='pt', padding_side='left', use_fast=False)

# gpt-2 doesnt have a padding token, so we have to add this
#tokenizer.pad_token = tokenizer.eos_token

random_gen = random.Random(42)

data_collator = DataCollatorForNI(tokenizer, num_pos_examples=0, text_only=True, max_source_length=768, max_target_length=128)
def convert_format(example):
        task_idx = re.findall(r"^task(\d+)_", example["Task"])
        assert len(task_idx) == 1
        task_idx = int(task_idx[0])
        processed_res = data_collator([example])
        #processed_res = { k: v.long().flatten().tolist() for k, v in processed_res.items() }
        # decoder-style input
        #res = {}
        #res['input_ids'] = processed_res['input_ids']
        # gpt2 expects causal training. we replace the instruction ids with -100 to mask it out in training.
        # res['labels'] = [-100] * len(processed_res['input_ids']) + processed_res['labels']
        #res['attention_mask'] = processed_res['attention_mask']
        return {
            "id": example["id"],
            "targets": random_gen.choice(example["Instance"]["output"]),
            "references": example["Instance"]["output"],
            "input": processed_res["inputs"][0]
            #**res
        }

ds = load_dataset("ni_dataset.py")['test']
original_columns = ds.column_names
ds = ds.map(convert_format)
ds.set_format("pt")

# generator = pipeline(
#     task="text-generation",
#     model=model,
#     tokenizer=tokenizer,
#     max_length=1024,
#     device_map="auto"
# )

ins = []
ids = []
for sample in ds:
    ins.append(sample['input'])
    ids.append(sample['id'])

outs = []
print('generating...')
for inputs in tqdm(KeyDataset(ds, "input"), total=len(ins)):
    inputs = tokenizer(inputs, return_tensors='pt').to(0)
    len_ids = inputs.input_ids.shape[-1]
    output_tokens = model.generate(**inputs, do_sample=False, max_length=1024)[0]
    output = tokenizer.decode(output_tokens[len_ids:], skip_special_tokens=True)
    outs.append(output)


res = []
for i, o in zip(ids, outs):
    res.append(json.dumps({'id': i, 'prediction': o}) + '\n')

with open('res.json', 'w') as w:
    w.writelines(res)
