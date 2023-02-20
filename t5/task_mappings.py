'''
Place all your tasks
'''
import abc
from datasets import load_dataset
from typing import Optional


task_mappings = {}


class AbstractTask(abc.ABC):
    name: str = NotImplemented
    subset_name: Optional[str] = None
    split_mapping = {
        "train": "train",
        "validation": "validation",
        "test": "test",
    }

    def load_dataset(self, split: int):
        return load_dataset(self.name, self.subset_name, split=split)

    def convert_seq2seq(self, example):
        raise NotImplementedError("Must be implemented by subclass")

    def load_seq2seq_dataset(self, split: str):
        dataset = self.load_dataset(self.split_mapping[split])
        return dataset.map(lambda x: self.convert_seq2seq(x))


# e.g., RTE
class RTE(AbstractTask):
    name = "super_glue"
    subset_name = "rte"

    def convert_seq2seq(self, example):
        return {
            "inputs": f'paraphrase: "{example["premise"]}" "{example["hypothesis"]}"',
            "targets": f"{example['label']}"
        }


task_mappings['rte'] = RTE()
