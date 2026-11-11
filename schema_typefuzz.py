"""Feed wrong-typed values into every numeric field of Soup's config models.

For each numeric field this tries a handful of values that are not a sane number
(a bool, a string, a list, None, an integer too big for a float) and records what
the model does with them.

  accepted   the model built and the bad value was quietly turned into a number
  rejected   pydantic.ValidationError, which is what we want
  escaped    any other exception, which would surface to the user as a traceback

By default every pydantic model defined in soup_cli.config.schema is probed. A model
with required fields gets a throwaway base value for each of them (a string, a list of
one string, 1, 1.0 or False), so DataConfig and friends are covered too.

Usage
    python schema_typefuzz.py path/to/Soup/src [--json out.json] [--markdown out.md]

The target is imported, not installed, so this only needs pydantic (plus whatever
soup_cli.config.schema pulls in at import time). Exit code is 1 while anything is
accepted or escaping, 0 when the schema is clean.
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


def throwaway(annotation):
    """A value that satisfies a required field, only so the model can be built."""
    origin = typing.get_origin(annotation)
    if origin is typing.Annotated:
        return throwaway(typing.get_args(annotation)[0])
    if origin in (typing.Union, getattr(types, "UnionType", None)):
        for arg in typing.get_args(annotation):
            if arg is not type(None):
                return throwaway(arg)
    if origin in (list, typing.List):
        return [throwaway(typing.get_args(annotation)[0])] if typing.get_args(annotation) else []
    if annotation is str:
        return "x"
    if annotation is bool:
        return False
    if annotation is int:
        return 1
    if annotation is float:
        return 1.0
    return None


def base_kwargs(model):
    """Defaults if the model builds on its own, else a throwaway value per required field, else None."""
    try:
        model()
        return {}
    except Exception:
        pass
    kwargs = {k: throwaway(v.annotation) for k, v in model.model_fields.items() if v.is_required()}
    try:
        model(**kwargs)
        return kwargs
    except Exception:
        return None


def discover(schema, pydantic_base):
    return [name for name, obj in vars(schema).items()
            if isinstance(obj, type) and issubclass(obj, pydantic_base) and obj is not pydantic_base
            and obj.__module__ == schema.__name__]


def probe_model(model, name, ValidationError):
    rows = []
    base = base_kwargs(model)
    if base is None:
        return rows, False
    for field, info in model.model_fields.items():
        kind = numeric_kind(info.annotation)
        if kind is None or field in base:
            continue
        for probe_name, value in PROBES.items():
            try:
                built = model(**{**base, field: value})
                outcome, detail = "accepted", repr(getattr(built, field))
            except ValidationError:
                outcome, detail = "rejected", ""
            except Exception as exc:
                outcome, detail = "escaped", f"{type(exc).__name__}: {str(exc)[:80]}"
            rows.append({"model": name, "field": field, "kind": kind, "probe": probe_name,
                         "outcome": outcome, "detail": detail})
    return rows, True


def markdown(rows, skipped):
    lines = ["| model | numeric fields | accept a bool | escape as a raw exception |", "|---|---|---|---|"]
    for name in sorted({r["model"] for r in rows}):
        mine = [r for r in rows if r["model"] == name]
        total = {r["field"] for r in mine}
        booly = {r["field"] for r in mine if r["probe"].startswith("bool") and r["outcome"] == "accepted"}
        esc = {r["field"] for r in mine if r["outcome"] == "escaped"}
        lines.append(f"| {name} | {len(total)} | {len(booly)} | {len(esc)} |")
    if skipped:
        lines += ["", "Skipped, could not be built even with throwaway values: " + ", ".join(sorted(skipped))]
    return "\n".join(lines) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src", help="the Soup src/ directory (contains soup_cli/)")
    ap.add_argument("--models", nargs="*", help="model names, default is every pydantic model in the module")
    ap.add_argument("--json", help="write the full result here")
    ap.add_argument("--markdown", help="write a per-model summary table here")
    args = ap.parse_args()

    sys.path.insert(0, str(Path(args.src).resolve()))
    schema = importlib.import_module("soup_cli.config.schema")
    from pydantic import BaseModel, ValidationError

    names = args.models or discover(schema, BaseModel)
    rows, skipped = [], []
    for name in names:
        model = getattr(schema, name, None)
        if model is None:
            print(f"skip {name}, not found", file=sys.stderr)
            continue
        found, built = probe_model(model, name, ValidationError)
        rows.extend(found)
        if not built:
            skipped.append(name)

    fields_bool = {(r["model"], r["field"]) for r in rows if r["probe"].startswith("bool") and r["outcome"] == "accepted"}
    fields_escape = {(r["model"], r["field"]) for r in rows if r["outcome"] == "escaped"}
    fields_total = {(r["model"], r["field"]) for r in rows}

    print(markdown(rows, skipped))
    print(f"numeric fields probed        {len(fields_total)}")
    print(f"fields that accept a bool    {len(fields_bool)}")
    print(f"fields where a probe escapes {len(fields_escape)}")
    print()
    for m, f in sorted(fields_escape):
        sample = next(r for r in rows if r["model"] == m and r["field"] == f and r["outcome"] == "escaped")
        print(f"escape  {m}.{f}  ({sample['probe']})  {sample['detail']}")

    if args.json:
        Path(args.json).write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")
    if args.markdown:
        Path(args.markdown).write_text(markdown(rows, skipped), encoding="utf-8")
    sys.exit(1 if (fields_bool or fields_escape) else 0)


if __name__ == "__main__":
    main()
