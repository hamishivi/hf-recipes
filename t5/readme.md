# T5 Seq2seq

T5 seq2seq training with arbitrary datasets and the ability to evaluate on multiple tasks at once. Install torch and the huggingface libraries (transformers, datasets, evaluate) and run e.g.:
```bash
python train.py --train_tasks rte --eval_tasks rte --output_dir test --model t5-small
```

Behind the scenes this is just a slight modification of the `Seq2SeqTrainer` and will accept all the [training arguments the huggingface trainers do](https://huggingface.co/docs/transformers/main_classes/trainer).

Distributed evaluation isn't guaranteed to work, but training should work fine.

There are some rough edges and bugs, I may or may not work these out.