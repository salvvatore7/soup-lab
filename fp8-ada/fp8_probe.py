"""Controlled FP8 step on one card. Uses Soup's own apply_fp8_training, then tries batch shapes.

case  = name,  fp8 conversion variant
shape = (batch, seq): tokens per step = batch * seq
"""
import json, sys, time, traceback
import torch
from transformers import AutoModelForCausalLM
from peft import LoraConfig, get_peft_model

NAME = "Qwen/Qwen2.5-0.5B-Instruct"


def build(variant):
    model = AutoModelForCausalLM.from_pretrained(NAME, dtype=torch.bfloat16).cuda()
    model.config.use_cache = False
    model = get_peft_model(model, LoraConfig(r=16, lora_alpha=32, target_modules=["q_proj", "v_proj"]))
    if variant == "soup":            # exactly what `quantization_aware: fp8` does
        from soup_cli.utils.fp8 import apply_fp8_training
        apply_fp8_training(model, "tensorwise")
    elif variant == "skip_lm_head":  # same recipe, lm_head left in bf16
        from torchao.float8 import convert_to_float8_training
        from torchao.float8.config import Float8LinearConfig
        convert_to_float8_training(model, config=Float8LinearConfig.from_recipe_name("tensorwise"),
                                   module_filter_fn=lambda m, fqn: "lm_head" not in fqn)
    elif variant != "bf16":
        raise ValueError(variant)
    return model


def one_step(variant, batch, seq):
    torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
    model = build(variant)
    ids = torch.randint(0, 151000, (batch, seq), device="cuda")
    torch.cuda.synchronize(); t = time.perf_counter()
    out = model(input_ids=ids, labels=ids)
    out.loss.backward()
    torch.cuda.synchronize(); dt = time.perf_counter() - t
    res = {"variant": variant, "batch": batch, "seq": seq, "tokens": batch * seq,
           "tokens_mod16": (batch * seq) % 16, "ok": True, "loss": round(out.loss.item(), 4),
           "peak_mib": round(torch.cuda.max_memory_allocated() / 2**20), "step_s": round(dt, 3)}
    del model, out
    return res


results = []
for variant in ["bf16", "soup", "skip_lm_head"]:
    for batch, seq in [(4, 64), (4, 86), (1, 12), (1, 16)]:
        try:
            r = one_step(variant, batch, seq)
        except Exception as exc:
            r = {"variant": variant, "batch": batch, "seq": seq, "tokens": batch * seq,
                 "tokens_mod16": (batch * seq) % 16, "ok": False,
                 "error": f"{type(exc).__name__}: {str(exc)[:160]}"}
        print(json.dumps(r), flush=True)
        results.append(r)
        torch.cuda.empty_cache()
json.dump({"gpu": torch.cuda.get_device_name(0), "capability": torch.cuda.get_device_capability(0),
           "torch": torch.__version__, "results": results}, open("fp8_probe.json", "w"), indent=1)
