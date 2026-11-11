"""Small checks that the two tools do what the README says. Run with pytest."""
import json
import subprocess
import sys
import typing
from pathlib import Path

import pytest

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))

import ablate  # noqa: E402
import schema_typefuzz as fuzz  # noqa: E402


def test_numeric_kind_sees_through_optional_and_annotated():
    assert fuzz.numeric_kind(int) == "int"
    assert fuzz.numeric_kind(float) == "float"
    assert fuzz.numeric_kind(typing.Optional[float]) == "float"
    assert fuzz.numeric_kind(typing.Annotated[int, "x"]) == "int"
    assert fuzz.numeric_kind(typing.Union[int, float]) == "float"
    assert fuzz.numeric_kind(bool) is None
    assert fuzz.numeric_kind(str) is None


def test_typefuzz_classifies_accepted_rejected_and_escaped(tmp_path):
    pkg = tmp_path / "soup_cli" / "config"
    pkg.mkdir(parents=True)
    (tmp_path / "soup_cli" / "__init__.py").write_text("")
    (pkg / "__init__.py").write_text("")
    (pkg / "schema.py").write_text(
        "from pydantic import BaseModel, field_validator\n"
        "class Toy(BaseModel):\n"
        "    lax: float = 1.0\n"  # takes True as 1.0, nothing stops it
        "class Bad(BaseModel):\n"
        "    rate: float = 1.0\n"
        "    @field_validator('rate', mode='before')\n"
        "    @classmethod\n"
        "    def _v(cls, v):\n"
        "        if isinstance(v, bool):\n"
        "            raise TypeError('rate must not be bool')\n"  # escapes pydantic
        "        return v\n"
        "class Good(BaseModel):\n"
        "    rate: float = 1.0\n"
        "    @field_validator('rate', mode='before')\n"
        "    @classmethod\n"
        "    def _v(cls, v):\n"
        "        if isinstance(v, bool):\n"
        "            raise ValueError('rate must not be bool')\n"  # becomes ValidationError
        "        return v\n"
    )
    out = tmp_path / "out.json"
    proc = subprocess.run(
        [sys.executable, str(HERE / "schema_typefuzz.py"), str(tmp_path), "--models", "Toy", "Bad", "Good",
         "--json", str(out)], capture_output=True, text=True)
    rows = json.loads(out.read_text())

    def outcome(model, probe):
        return next(r["outcome"] for r in rows if r["model"] == model and r["probe"] == probe)

    assert outcome("Toy", "bool_true") == "accepted"
    assert outcome("Bad", "bool_true") == "escaped"
    assert outcome("Good", "bool_true") == "rejected"
    assert proc.returncode == 1  # findings present


def test_schedule_gives_abba_for_two_rounds():
    assert "".join(ablate.schedule(["A", "B"], 2)) == "ABBA"
    assert "".join(ablate.schedule(["A", "B"], 4)) == "ABBAABBA"
    assert "".join(ablate.schedule(["A", "B", "C"], 2)) == "ABCCBA"


def test_summarise_counts_failures_and_spread():
    runs = [
        {"seconds": 10.0, "returncode": 0, "gpu_peak_mib": 100},
        {"seconds": 12.0, "returncode": 0, "gpu_peak_mib": 140},
        {"seconds": 1.0, "returncode": 1, "gpu_peak_mib": None},
    ]
    s = ablate.summarise(runs)
    assert s["runs"] == 3 and s["failed"] == 1
    assert s["median_s"] == 11.0 and s["spread_pct"] == pytest.approx(18.2, abs=0.1)
    assert s["gpu_peak_mib"] == 140


def test_ablate_runs_real_processes_and_writes_json(tmp_path):
    out = tmp_path / "r.json"
    py = sys.executable.replace("\\", "/")
    proc = subprocess.run(
        [sys.executable, str(HERE / "ablate.py"), "--rounds", "2", "--out", str(out),
         "--variant", f'fast="{py}" -c pass',
         "--variant", f'slow="{py}" -c "import time; time.sleep(0.3)"'],
        capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr
    data = json.loads(out.read_text())
    assert data["order"] == ["fast", "slow", "slow", "fast"]
    assert data["summary"]["slow"]["median_s"] > data["summary"]["fast"]["median_s"]


def test_typefuzz_discovers_models_and_builds_ones_with_required_fields(tmp_path):
    pkg = tmp_path / "soup_cli" / "config"
    pkg.mkdir(parents=True)
    (tmp_path / "soup_cli" / "__init__.py").write_text("")
    (pkg / "__init__.py").write_text("")
    (pkg / "schema.py").write_text(
        "from typing import List\n"
        "from pydantic import BaseModel\n"
        "class Needs(BaseModel):\n"
        "    train: List[str]\n"      # required, gets a throwaway value
        "    limit: int = 5\n"
        "class NotAModel:\n"
        "    limit = 1\n"
    )
    out, md = tmp_path / "o.json", tmp_path / "o.md"
    proc = subprocess.run(
        [sys.executable, str(HERE / "schema_typefuzz.py"), str(tmp_path), "--json", str(out), "--markdown", str(md)],
        capture_output=True, text=True)
    rows = json.loads(out.read_text())
    assert {r["model"] for r in rows} == {"Needs"}
    assert {r["field"] for r in rows} == {"limit"}  # `train` is only scaffolding, not probed
    assert any(r["probe"] == "bool_true" and r["outcome"] == "accepted" for r in rows)
    assert "| Needs | 1 | 1 | 0 |" in md.read_text()
    assert proc.returncode == 1
