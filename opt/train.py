import re
import torch
import random
from transformers import GPT2LMHeadModel, AutoTokenizer, DataCollatorForSeq2Seq
from transformers import OPTForCausalLM
from transformers import Seq2SeqTrainingArguments, Seq2SeqTrainer, logging
from datasets import load_dataset
from ni_collator import DataCollatorForNI
from evaluation import compute_metrics

# model = GPT2LMHeadModel.from_pretrained('gpt2-xl')
# tokenizer = AutoTokenizer.from_pretrained('gpt2-xl', return_tensors='pt')

model = OPTForCausalLM.from_pretrained('facebook/opt-13b')
tokenizer = AutoTokenizer.from_pretrained('facebook/opt-13b', return_tensors='pt', use_fast=False)

# gpt-2 doesnt have a padding token, so we have to add this
#tokenizer.pad_token = tokenizer.eos_token

random_gen = random.Random(42)

data_collator = DataCollatorForNI(tokenizer, num_pos_examples=2, max_source_length=768, max_target_length=128)
def convert_format(example):
        task_idx = re.findall(r"^task(\d+)_", example["Task"])
        assert len(task_idx) == 1
        task_idx = int(task_idx[0])
        processed_res = data_collator([example])
        processed_res = { k: v.long().flatten().tolist() for k, v in processed_res.items() }
        # if gpt and opt:
        # tricky detail - gpt2 tokenizer doesnt add EOS so we have to add it.
        # otherwise generation will never stop 😱
        # if opt: bos token is eos token, so remove from start of labels
        processed_res['labels'] = processed_res['labels'][1:] + [tokenizer.eos_token_id]
        # decoder-style input
        res = {}
        res['input_ids'] = processed_res['input_ids'] + processed_res['labels']
        
        # gpt2 expects causal training. we replace the instruction ids with -100 to mask it out in training.
        res['labels'] = [-100] * len(processed_res['input_ids']) + processed_res['labels']
        res['attention_mask'] = processed_res['attention_mask'] + [1]*len(processed_res['labels'])
        return {
            "id": example["id"],
            "targets": random_gen.choice(example["Instance"]["output"]),
            "references": example["Instance"]["output"],
            **res
        }

ds = load_dataset("ni_dataset.py")
original_columns = ds.column_names['train']
ds = ds.map(convert_format)
ds.set_format("pt")

training_args = Seq2SeqTrainingArguments(
    output_dir='trained_models/opt13b_2pos',
    per_device_train_batch_size=1,
    gradient_accumulation_steps=16,
    learning_rate=1e-5,
    warmup_steps=1000,
    overwrite_output_dir=True,
    do_train=True,
    bf16=True,
    evaluation_strategy="no",
    num_train_epochs=3,
    save_strategy="epoch",
    fsdp="full_shard auto_wrap",
    fsdp_transformer_layer_cls_to_wrap="OPTDecoderLayer"
)
trainer = Seq2SeqTrainer(
    model=model,
    tokenizer=tokenizer,
    args=training_args,
    train_dataset=ds['train'],
    eval_dataset=ds['test'],
    compute_metrics=compute_metrics,
    data_collator=DataCollatorForSeq2Seq(tokenizer)
)

result = trainer.train()
print(result)
