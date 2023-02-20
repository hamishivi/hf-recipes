import math
import time
from typing import Dict, List, Optional

import datasets
from torch.utils.data import DataLoader, Dataset
from transformers import Seq2SeqTrainer, TrainerControl, is_datasets_available
from transformers.trainer_utils import speed_metrics


class MultiEvalSeq2SeqTrainer(Seq2SeqTrainer):
    def __init__(
        self,
        *,
        eval_datasets: Optional[List[Dataset]] = None,
        eval_dataset_names: List[str] = None,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.eval_datasets = eval_datasets
        self.eval_dataset_names = eval_dataset_names

    def get_eval_dataloaders(
        self, eval_datasets: Optional[List[Dataset]] = None
    ) -> List[DataLoader]:
        """
        Returns a list of evaluation [`~torch.utils.data.DataLoader`].

        Args:
            eval_dataset (`torch.utils.data.Dataset`, *optional*):
                If provided, will override `self.eval_dataset`. If it is an `datasets.Dataset`,
                columns not accepted by the `model.forward()` method are automatically removed.
                It must implement `__len__`.
        """
        if eval_datasets is None and self.eval_dataset is None:
            raise ValueError("Trainer: evaluation requires an eval_dataset.")
        eval_datasets = eval_datasets if eval_datasets is not None else self.eval_datasets

        if not isinstance(eval_datasets, list):
            eval_datasets = [eval_datasets]

        if is_datasets_available() and isinstance(eval_datasets[0], datasets.Dataset):
            eval_datasets = [
                self._remove_unused_columns(dataset, description="evaluation")
                for dataset in eval_datasets
            ]
        else:
            raise ValueError("Trainer: evaluation requires a list of eval_datasets")

        eval_samplers = [self._get_eval_sampler(eval_dataset) for eval_dataset in eval_datasets]

        return [
            DataLoader(
                eval_dataset,
                sampler=eval_sampler,
                batch_size=self.args.eval_batch_size,
                collate_fn=self.data_collator,
                drop_last=self.args.dataloader_drop_last,
                num_workers=self.args.dataloader_num_workers,
                pin_memory=self.args.dataloader_pin_memory,
            )
            for eval_sampler, eval_dataset in zip(eval_samplers, eval_datasets)
        ]

    def evaluate(
        self,
        eval_datasets: Optional[List[Dataset]] = None,
        ignore_keys: Optional[List[str]] = None,
        metric_key_prefix: str = "eval",
        **gen_kwargs
    ) -> Dict[str, float]:
        """
        Run evaluation and returns metrics.

        The calling script will be responsible for providing
        a method to compute metrics, as they are task-dependent
        (pass it to the init `compute_metrics` argument).

        You can also subclass and override this method to inject custom behavior.

        Args:
            eval_datasets (`Dataset`, *optional*):
                Pass a dataset if you wish to override `self.eval_dataset`.
                If it is an `datasets.Dataset`, columns not
                accepted by the `model.forward()` method are automatically removed.
                It must implement the `__len__` method.
            ignore_keys (`List[str]`, *optional*):
                A list of keys in the output of your model (if it is a dictionary)
                that should be ignored when gathering predictions.
            metric_key_prefix (`str`, *optional*, defaults to `"eval"`):
                An optional prefix to be used as the metrics key prefix.
                For example the metrics "bleu" will be named "eval_bleu" if
                the prefix is `"eval"` (default)
            max_length (`int`, *optional*):
                The maximum target length to use when predicting with the
                generate method.
            num_beams (`int`, *optional*):
                Number of beams for beam search that will be used when predicting
                with the generate method. 1 means no beam search.

        Returns:
            A dictionary containing the evaluation loss and the potential metrics
            computed from the predictions. The dictionary also contains the epoch
            number which comes from the training state.
        """
        gen_kwargs = gen_kwargs.copy()
        if gen_kwargs.get("max_length") is None and gen_kwargs.get("max_new_tokens") is None:
            gen_kwargs["max_length"] = self.args.generation_max_length
        gen_kwargs["num_beams"] = (
            gen_kwargs["num_beams"] if gen_kwargs.get("num_beams") is not None else self.args.generation_num_beams
        )
        self._gen_kwargs = gen_kwargs
        self._memory_tracker.start()

        assert (
            self.eval_dataset_names is not None
        ), "Must have eval dataset names to do proper eval."

        eval_dataloaders = self.get_eval_dataloaders(eval_datasets)
        start_time = time.time()

        output_metrics = {}
        for dataset_name, eval_dataloader in zip(self.eval_dataset_names, eval_dataloaders):
            alt_metric_key_prefix = dataset_name + "." + metric_key_prefix
            eval_loop = (
                self.prediction_loop
                if self.args.use_legacy_prediction_loop
                else self.evaluation_loop
            )
            output = eval_loop(
                eval_dataloader,
                description="Evaluation",
                # No point gathering the predictions if there are no metrics, otherwise we defer to
                # self.args.prediction_loss_only
                prediction_loss_only=True if self.compute_metrics is None else None,
                ignore_keys=ignore_keys,
                metric_key_prefix=alt_metric_key_prefix,
            )
            output_metrics.update(output.metrics)

            total_batch_size = self.args.eval_batch_size * self.args.world_size
            output_metrics.update(
                speed_metrics(
                    alt_metric_key_prefix,
                    start_time,
                    num_samples=output.num_samples,
                    num_steps=math.ceil(output.num_samples / total_batch_size),
                )
            )

        self.log(output_metrics)

        self.control: TrainerControl = self.callback_handler.on_evaluate(
            self.args, self.state, self.control, output.metrics
        )

        self._memory_tracker.stop_and_update_metrics(output.metrics)

        return output_metrics
