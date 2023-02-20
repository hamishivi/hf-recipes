import json
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import evaluate
import numpy as np
from datasets import interleave_datasets, load_dataset
from multi_eval_seq2seq_trainer import MultiEvalSeq2SeqTrainer
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    HfArgumentParser,
    Seq2SeqTrainingArguments,
)
from task_mappings import task_mappings


@dataclass
class DataArguments:
    """
    Arguments about training, not covered by the seq2seq arguments
    """

    train_tasks: List[str] = field(
        default_factory=list,
        metadata={
            "help": f"List of the tasks to train on. Tasks must be in {task_mappings.keys()}."
        },
    )
    eval_tasks: List[str] = field(
        default_factory=list,
        metadata={
            "help": f"List of the tasks to evaluate on. Tasks must be from {task_mappings.keys()}."
        },
    )
    max_samples_per_train_dataset: Optional[int] = field(
        default=-1,
        metadata={"help": "Max instances to take from any train prompt."},
    )
    max_samples_per_eval_dataset: Optional[int] = field(
        default=-1, metadata={"help": "Max instances to take from any eval prompt."}
    )
    model_name: str = field(
        default="google/t5-xl-lm-adapt",
        metadata={"help": "Name of model. Must be a AutoModelForSeq2SeqLM-compatible model."},
    )
    tokenizer_name: Optional[str] = field(
        default=None,
        metadata={"help": "Name of tokenizer to use. If not given, assume same name as model."},
    )
    metrics_output: str = field(
        default="metrics.json",
        metadata={"help": "Name of file to output metrics too. Default: metrics.json"},
    )
    max_source_length: int = field(
        default=768,
        metadata={"help": "Maximum length of inputs."},
    )
    max_target_length: int = field(
        default=256,
        metadata={"help": "Maximum length of outputs and generated text."},
    )


parser = HfArgumentParser((Seq2SeqTrainingArguments, DataArguments))
training_args, data_args = parser.parse_args_into_dataclasses()


train_tasks = data_args.train_tasks
eval_tasks = data_args.eval_tasks

train_datasets = []
for task in train_tasks:
    if task not in task_mappings:
        raise ValueError(
            f"train task {task} not valid. Tasks must be from {task_mappings.keys()}"
        )
    train_datasets.append(task_mappings[task].load_seq2seq_dataset("train"))
    # cap at task level so we have roughly similar amounts of training data.
    if (
        data_args.max_samples_per_train_dataset > 0
        and len(train_datasets[-1]) > data_args.max_samples_per_train_dataset
    ):
        train_datasets[-1] = train_datasets[-1].select(
            range(data_args.max_samples_per_train_dataset)
        )

eval_datasets = []
eval_dataset_names = []
for task in eval_tasks:
    if task not in task_mappings:
        raise ValueError(f"eval task {task} not valid. Tasks must be from {task_mappings.keys()}")
    eval_datasets.append(task_mappings[task].load_seq2seq_dataset("validation"))
    eval_dataset_names.append(task)

tokenizer_name = (
    data_args.tokenizer_name if data_args.tokenizer_name is not None else data_args.model_name
)
tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
model = AutoModelForSeq2SeqLM.from_pretrained(data_args.model_name)


def preprocess_function(example):
    output = tokenizer(example['inputs'], truncation=True, max_length=data_args.max_source_length)
    output['labels'] = tokenizer(example['targets'], truncation=True, max_length=data_args.max_target_length)['input_ids']
    return output


train_datasets = [ds.map(preprocess_function, remove_columns=ds.column_names) for ds in train_datasets]
eval_datasets = [ds.map(preprocess_function, remove_columns=ds.column_names) for ds in eval_datasets]

metric = evaluate.load("accuracy")


def compute_metrics(eval_preds):
    preds, labels = eval_preds
    # decode preds and labels
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    result = metric.compute(predictions=decoded_preds, references=decoded_labels)
    return result


trainer = MultiEvalSeq2SeqTrainer(
    model=model,
    tokenizer=tokenizer,
    args=training_args,
    data_collator=DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
    ),
    train_dataset=interleave_datasets(train_datasets, seed=training_args.seed),
    eval_datasets=eval_datasets,
    eval_dataset=eval_datasets,
    eval_dataset_names=eval_dataset_names,
    compute_metrics=compute_metrics,
)

print("Training model!")
try:
    output = trainer.train(resume_from_checkpoint=training_args.output_dir)
except ValueError:
    output = trainer.train()

print("Evaluating model!")
metrics = trainer.evaluate(eval_datasets=eval_datasets, max_length=data_args.max_target_length)

print('Results:')
print(metrics)

# save to metrics.json for beaker :)
print('Saving results to', data_args.metrics_output)
with open(data_args.metrics_output, "w") as w:
    json.dump(metrics, w)
