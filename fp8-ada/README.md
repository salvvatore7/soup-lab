# FP8 training on an Ada card (sm_89), for Soup issue #835 item 4

Hardware and stack, as `soup doctor` and `pip list` printed them.

| | |
|---|---|
| Card | NVIDIA GeForce RTX 4060 Ti, 8 GB, compute capability 8.9 (Ada) |
| Driver | 610.88, CUDA 13.0 runtime |
| OS, Python | Windows 11, Python 3.13.0 |
| torch, torchao | 2.14.1+cu130, 0.18.0 |
| transformers, peft, trl | 5.22.2, 0.21.2, 0.29.1 |
| Soup | 0.77.2, source checkout at `2db8e0c` |
| `soup doctor` | `FP8: hardware=yes, software=yes` |

## 1. A real `soup train` with `quantization_aware: fp8`

Config is [fp8.yaml](fp8.yaml): Qwen2.5-0.5B-Instruct, LoRA r=16 on `q_proj` and `v_proj`, `fp8_recipe: tensorwise`, batch size 4, the 10-row `alpaca_tiny.jsonl` from `examples/data`.

On unmodified `2db8e0c`, `soup train --config fp8.yaml --yes` prints `FP8 training enabled (Float8Linear, recipe=tensorwise)`, applies LoRA, and dies on the first step.

```
RuntimeError: Expected trailing dimension of mat1 to be divisible by 16
but got mat1 shape: (151936x344).
```

344 is the token count of that batch, and 151936 is the vocabulary, so the failing GEMM is the one at the LM head.

With the two-line change in [pad_to_multiple_of_16.patch](pad_to_multiple_of_16.patch) (set `pad_to_multiple_of = 16` on the SFT arguments when `quantization_aware == "fp8"`) the same command completes 3 of 3 steps. Train loss 1.81 over the run, eval loss 2.453, GPU peak 2.8 of 8.0 GB, adapter saved.

## 2. Which shapes fail

[fp8_probe.py](fp8_probe.py) builds the model, applies LoRA, calls Soup's own `apply_fp8_training`, and runs one forward and backward on random token ids. Results are in [fp8_probe.json](fp8_probe.json). The ids are random, so the losses mean nothing, and step times include first-call warm-up, so they are not a speed comparison.

| batch x seq | tokens | tokens mod 16 | bf16 | Soup FP8 | FP8 with `lm_head` skipped |
|---|---|---|---|---|---|
| 4 x 64 | 256 | 0 | ok | ok | ok |
| 4 x 86 | 344 | 8 | ok | fails (151936x344) | fails (896x344) |
| 1 x 12 | 12 | 12 | ok | fails (151936x12) | fails (896x12) |
| 1 x 16 | 16 | 0 | ok | ok | ok |

Every shape with a token count divisible by 16 passes and every other one fails. Skipping `lm_head` in the conversion only moves the failure to the next layer, so a filter on the vocabulary layer is not enough and the batch itself has to be padded.

## Not covered

Only the `tensorwise` recipe, which is the only one that can run on Windows (`rowwise` is never built there). Only the SFT trainer, the patch does not touch the DPO family or the other trainers that call `apply_fp8_training`. One model, one card, one short run. Nothing here says anything about speed or about loss quality over a real training run.
