"""Feed wrong-typed values into every numeric field of Soup's config models.

For each numeric field this tries a handful of values that are not a sane number
(a bool, a string, a list, None, an integer too big for a float) and records what
the model does with them.

  accepted   the model built and the bad value was quietly turned into a number
  rejected   pydantic.ValidationError, which is what we want
  escaped    any other exception, which would surface to the user as a traceback

Usage
    python schema_typefuzz.py path/to/Soup/src [--json out.json]

The target is imported, not installed, so this only needs pydantic (plus whatever
soup_cli.config.schema pulls in at import time).
"""
import argparse
import importlib
import json
import sys
import types
import typing
from pathlib import Path

PROBES = {
    "bool_true": True,
    "bool_false": False,
    "str": "abc",
    "list": [1],
    "none": None,
    "huge_int": 10 ** 400,
}


def numeric_kind(annotation):
    """Return 'int', 'float' or None for a field annotation, looking through Optional/Union/Annotated."""
    origin = typing.get_origin(annotation)
    if origin is typing.Annotated:
        return numeric_kind(typing.get_args(annotation)[0])
    if origin in (typing.Union, getattr(types, "UnionType", None)):
        kinds = {numeric_kind(a) for a in typing.get_args(annotation) if a is not type(None)}
        kinds.discard(None)
        if kinds == {"int"}:
            return "int"
        if kinds and kinds <= {"int", "float"}:
            return "float"
        return None
    if annotation is bool:
        return None
    if annotation is int:
        return "int"
    if annotation is float:
        return "float"
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src", help="the Soup src/ directory (contains soup_cli/)")
    ap.add_argument("--models", nargs="*", default=["LoraConfig", "TrainingConfig", "DataConfig", "EvalGateConfig"])
    ap.add_argument("--json", help="write the full result here")
    args = ap.parse_args()

    sys.path.insert(0, str(Path(args.src).resolve()))
    schema = importlib.import_module("soup_cli.config.schema")
    from pydantic import ValidationError

    rows = []
    for name in args.models:
        model = getattr(schema, name, None)
        if model is None:
            print(f"skip {name}, not found", file=sys.stderr)
            continue
        try:
            model()
        except Exception as exc:  # a model with required fields needs a hand-written base
            print(f"skip {name}, cannot build with defaults ({type(exc).__name__})", file=sys.stderr)
            continue
        for field, info in model.model_fields.items():
            kind = numeric_kind(info.annotation)
            if kind is None:
                continue
            for probe_name, value in PROBES.items():
                try:
                    built = model(**{field: value})
                    outcome, detail = "accepted", repr(getattr(built, field))
                except ValidationError:
                    outcome, detail = "rejected", ""
                except Exception as exc:
                    outcome, detail = "escaped", f"{type(exc).__name__}: {str(exc)[:80]}"
                rows.append({"model": name, "field": field, "kind": kind, "probe": probe_name,
                             "outcome": outcome, "detail": detail})

    def count(pred):
        return sum(1 for r in rows if pred(r))

    fields_bool = {(r["model"], r["field"]) for r in rows if r["probe"].startswith("bool") and r["outcome"] == "accepted"}
    fields_escape = {(r["model"], r["field"]) for r in rows if r["outcome"] == "escaped"}
    fields_total = {(r["model"], r["field"]) for r in rows}

    print(f"numeric fields probed        {len(fields_total)}")
    print(f"fields that accept a bool    {len(fields_bool)}")
    print(f"fields where a probe escapes {len(fields_escape)}")
    for probe_name in PROBES:
        acc = count(lambda r: r["probe"] == probe_name and r["outcome"] == "accepted")
        esc = count(lambda r: r["probe"] == probe_name and r["outcome"] == "escaped")
        print(f"  {probe_name:10s} accepted {acc:3d}   escaped {esc:3d}")
    print()
    for m, f in sorted(fields_escape):
        sample = next(r for r in rows if r["model"] == m and r["field"] == f and r["outcome"] == "escaped")
        print(f"escape  {m}.{f}  ({sample['probe']})  {sample['detail']}")

    if args.json:
        Path(args.json).write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")
    sys.exit(1 if (fields_bool or fields_escape) else 0)


if __name__ == "__main__":
    main()
