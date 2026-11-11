# soup-lab

Two small tools I wrote while reading [Soup](https://github.com/MakazhanAlpamys/Soup) on a laptop-class card (RTX 4060 Ti, 8 GB, 32 GB RAM). Both are standard library plus pydantic, no torch needed.

## schema_typefuzz.py

Feeds wrong-typed values into every numeric field of Soup's config models and says what happens. Each field gets a `True`, a `False`, a string, a list, `None` and an integer too big for a float. The outcome is one of three.

- accepted, the bad value was quietly turned into a number
- rejected, pydantic raised `ValidationError`, which is the good case
- escaped, some other exception got out, which the user sees as a raw traceback

```
python schema_typefuzz.py path/to/Soup/src --json out.json
```

### What it found on Soup at commit 2db8e0c (2026-11-11)

Models probed were `LoraConfig`, `TrainingConfig` and `EvalGateConfig`. `DataConfig` has required arguments, so it needs a hand-written base and is skipped for now.

| | count |
|---|---|
| numeric fields probed | 117 |
| fields that accept a `bool` | 55 |
| fields where a probe escapes as a raw exception | 11 |

Examples from the full list in [results/soup-2db8e0c.json](results/soup-2db8e0c.json). `lr: true` becomes a learning rate of 1.0 and `lora.r: false` builds a rank 0 adapter. `num_labels: true`, `distill_temperature: true` and `unlearn_alpha: true` raise `TypeError` out of the schema, and `grpo_beta` with an integer over 10^308 raises `OverflowError`.

### Why the escapes happen

Pydantic v2 turns only `ValueError` and `AssertionError` raised inside a validator into a `ValidationError`. Anything else propagates. Many of Soup's field validators call helpers that `raise TypeError` for a bool (`utils/ebft_gdpo.py` is one), and a few raise it directly in `config/schema.py`. A grep finds about 1,250 `raise TypeError` in `soup_cli/utils`, so it looks like a house convention rather than a slip. The tests that cover these helpers pass, because they call the helper directly and accept either `TypeError` or `ValueError` (see `tests/test_v0520.py` for `validate_ebft_temperature`). Only the path through the schema is wrong.

The bool acceptance has the same shape. Fields that have a hand-written bool guard reject it and the rest fall back to pydantic's lax mode, which is happy to read `True` as `1.0`.

### A fix that does not touch 1,250 sites

Wrap the validators once. A small decorator that catches `TypeError` and `OverflowError` and re-raises `ValueError` keeps the field name in the message, because pydantic adds the location itself. For the 55 lax fields, a shared numeric type that rejects bool (`Annotated[float, AfterValidator(...)]` or `StrictFloat`) covers them in one place. This tool is the regression test for both, it exits 1 while anything is accepted or escaping, and 0 when the schema is clean.

## ablate.py

Runs command variants in an interleaved order (A B B A, then A B B A again) so thermal throttling or a warming cache cannot land on one variant only. Every run is a fresh process. It records wall time, and with `--gpu` the peak GPU memory sampled from `nvidia-smi`. Meant for comparing things like `stream_buffers` values on a real card.

```
python ablate.py --rounds 2 --gpu \
  --variant buffers2="soup train --config sft.yaml" \
  --variant buffers3="soup train --config sft.yaml" \
  --out ablation.json
```

I have only tested the harness itself so far, with throwaway commands. I have not yet run Soup through it, and I will post numbers from the 4060 Ti when I have them.

## Tests

```
pip install -r requirements.txt
pytest -q
```

Five tests. They build a toy `soup_cli.config.schema` with one lax field, one that raises `TypeError` and one that raises `ValueError`, and check the three outcomes. They also check the A B B A order, the summary maths and one real end-to-end run of the harness.

## Limits

The fuzz only probes numeric fields and only the models listed above. It does not run cross-field validators or the top-level `SoupConfig`. A clean result means these probes are handled, not that the schema is bug free.
