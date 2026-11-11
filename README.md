# soup-lab

Two small tools I wrote while reading [Soup](https://github.com/MakazhanAlpamys/Soup) on a laptop-class card (RTX 4060 Ti, 8 GB, 32 GB RAM). Standard library plus pydantic, no torch needed. Apache-2.0, same as Soup.

## schema_typefuzz.py

Feeds wrong-typed values into every numeric field of Soup's config models and says what happens. Each field gets `True`, `False`, a string, a list, `None` and an integer too big for a float. The outcome is one of three.

- accepted, the bad value was quietly turned into a number
- rejected, pydantic raised `ValidationError`, which is the good case
- escaped, some other exception got out, which the user sees as a raw traceback

```
python schema_typefuzz.py path/to/Soup/src --json out.json --markdown out.md
```

It imports the target instead of installing it. By default it probes every pydantic model defined in `soup_cli.config.schema`. A model with required fields gets a throwaway value for each of them, so `DataConfig` is covered. The exit code is 1 while anything is accepted or escaping and 0 when the schema is clean, so it can run in CI.

### Result on Soup at 2db8e0c (2026-11-11)

| model | numeric fields | accept a bool | escape as a raw exception |
|---|---|---|---|
| DataConfig | 11 | 6 | 0 |
| EvalGateConfig | 2 | 2 | 0 |
| LoraConfig | 4 | 4 | 0 |
| ShipConfig | 2 | 1 | 0 |
| TrainingConfig | 111 | 49 | 11 |

130 numeric fields in total, 62 accept a bool, 11 escape. `SoupConfig` itself cannot be built from throwaway values and is skipped. The full list is in [results/soup-2db8e0c.json](results/soup-2db8e0c.json).

What that means, and what is already being handled upstream.

- `lr: true` becoming a learning rate of 1.0 is [#2375](https://github.com/MakazhanAlpamys/Soup/issues/2375), with a fix in PR #2462 for `LoraConfig` and `TrainingConfig`. The probe reproduces the numbers from that issue.
- The 11 escapes are all in `TrainingConfig` and are [#2366](https://github.com/MakazhanAlpamys/Soup/issues/2366), with a fix in PR #2382. The cause is that pydantic v2 turns only `ValueError` and `AssertionError` raised in a validator into a `ValidationError`, while helpers such as `validate_ebft_temperature` raise `TypeError`.
- What I could not find in the tracker is the same bool acceptance in `DataConfig`, `EvalGateConfig` and `ShipConfig` (nine fields, for example `eval.ship.forgetting_threshold: true` reads as 1.0). I filed that as [#2712](https://github.com/MakazhanAlpamys/Soup/issues/2712).

## ablate.py

Runs command variants in an interleaved order (A B B A, then A B B A again) so thermal throttling or a warming cache cannot land on one variant only. Every run is a fresh process. It records wall time, and with `--gpu` the peak GPU memory sampled from `nvidia-smi`. Meant for comparing things like `stream_buffers` values on a real card.

```
python ablate.py --rounds 2 --gpu \
  --variant buffers2="soup train --config sft.yaml" \
  --variant buffers3="soup train --config sft.yaml" \
  --out ablation.json
```

So far I have only tested the harness itself, with throwaway commands. I have not run Soup through it yet, and numbers from the 4060 Ti will land in `results/` when I have them.

## Tests

```
pip install -r requirements.txt
pytest -q
```

They build a toy `soup_cli.config.schema` with a lax field, a field that raises `TypeError`, a field that raises `ValueError` and a model with a required field, and check the outcomes. They also check the A B B A order, the summary maths and one real end-to-end run of the harness. CI runs them on Python 3.11 and 3.13.

## Limits

The fuzz only probes numeric fields. It does not run cross-field validators or the top-level `SoupConfig`. A clean result means these probes are handled, not that the schema is bug free. It also reports fields that construct, not fields that behave badly, so a few hits may be intentional.
