import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from ecgtrust import render, synth  # noqa: E402


@pytest.fixture(scope="session")
def sig():
    return synth.synthetic_signal(seed=3)


@pytest.fixture(scope="session")
def rendered(sig):
    return render.render_ecg(sig.signal_mv, sig.fs, render.RenderConfig())


@pytest.fixture(scope="session")
def quick_data_available():
    return (ROOT / "data" / "quick" / "ptbxl_database.csv").exists() or any((ROOT / "data" / "quick").rglob("*.dat"))


@pytest.fixture(scope="session")
def frozen_model_path():
    p = ROOT / "results" / "trust_model.json"
    if not p.exists():
        pytest.skip("results/trust_model.json not generated yet (run scripts/run_evaluation.py --preset full)")
    return p
