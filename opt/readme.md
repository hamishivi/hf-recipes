# Training OPT Models

A quick and dirty repo for training opt models on SNI. Requires accelerate, transformers, datasets, pytorch.

Not at all guaranteed to be correct.

Adjust training script and run training with (adjust nodes):
```
python -m torch.distributed.launch --nproc_per_node=8 train.py
```

To generate on eval:
```
python generate.py
```

To run eval script:
```
python evaluation.py --prediction_file res.json --reference_file test_references.jsonl
```
