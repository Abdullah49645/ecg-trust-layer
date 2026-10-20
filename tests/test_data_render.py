import hashlib

import numpy as np
import pytest

from ecgtrust import data, degrade, render


def test_select_records_deterministic_one_per_patient():
    meta = [{"ecg_id": str(i), "patient_id": str(i // 2), "filename_hr": f"records500/{i:05d}_hr", "strat_fold": str(1 + i % 10)} for i in range(400)]
    a = data.select_records(meta, {"tune": 10, "val": 5, "test": 8}, seed=1)
    b = data.select_records(meta, {"tune": 10, "val": 5, "test": 8}, seed=1)
    assert a == b
    for split, rows in a.items():
        pats = [r["patient_id"] for r in rows]
        assert len(pats) == len(set(pats))
        folds = {int(float(r["strat_fold"])) for r in rows}
        assert folds <= set(data.SPLIT_FOLDS[split])
    # no patient shared across splits (official folds keep a patient in one fold)
    allp = [r["patient_id"] for rows in a.values() for r in rows]
    assert len(allp) == len(set(allp))


def test_load_metadata_missing_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        data.load_metadata(tmp_path)


def test_load_record_missing_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        data.load_record(tmp_path, "records500/00000/00001_hr", "II")


def test_load_real_record_if_present(quick_data_available):
    import json
    from pathlib import Path

    sel_path = Path(__file__).resolve().parents[1] / "configs" / "quick_selection.json"
    if not quick_data_available or not sel_path.exists():
        pytest.skip("bundled quick records not present")
    row = data.load_selection(sel_path)["test"][0]
    sig = data.load_record(Path(__file__).resolve().parents[1] / "data" / "quick", row["filename_hr"], "II", row)
    assert sig.fs == 500.0 and sig.lead.upper() == "II" and sig.units == "mV"
    assert sig.duration_s == pytest.approx(10.0, abs=0.01)
    assert np.all(np.isfinite(sig.signal_mv))
    with pytest.raises(ValueError):
        data.load_record(Path(__file__).resolve().parents[1] / "data" / "quick", row["filename_hr"], "NOT_A_LEAD", row)


def test_render_geometry_and_calibration(sig):
    cfg = render.RenderConfig(px_per_mm=8.0)
    r = render.render_ecg(sig.signal_mv, sig.fs, cfg)
    h, w = r.image.shape[:2]
    assert w == int(round((10 * 25 + 12) * 8)) and h == int(round((80 + 12) * 8))
    # time mapping: 1 s == 25 mm == 200 px
    assert r.x_of_t(np.array([1.0]))[0] - r.x_of_t(np.array([0.0]))[0] == pytest.approx(200.0)
    # amplitude mapping: 1 mV == 10 mm == 80 px (upwards is smaller y)
    assert r.y_of_mv(np.array([0.0]))[0] - r.y_of_mv(np.array([1.0]))[0] == pytest.approx(80.0)


def test_render_is_deterministic(sig):
    a = render.render_ecg(sig.signal_mv, sig.fs, render.RenderConfig())
    b = render.render_ecg(sig.signal_mv, sig.fs, render.RenderConfig())
    assert hashlib.sha256(a.image.tobytes()).hexdigest() == hashlib.sha256(b.image.tobytes()).hexdigest()


def test_render_config_roundtrip():
    c = render.RenderConfig(px_per_mm=6.0, trace_thickness_mm=0.5)
    assert render.RenderConfig.from_dict(c.to_dict()) == c


@pytest.mark.parametrize("bad", [np.array([]), np.array([1.0]), np.array([0.0, np.nan, 1.0])])
def test_render_rejects_invalid_signal(bad):
    with pytest.raises(ValueError):
        render.render_ecg(bad, 500.0)


def test_render_rejects_bad_fs(sig):
    with pytest.raises(ValueError):
        render.render_ecg(sig.signal_mv, 0.0)


def test_degradation_reproducible_and_parameters_recorded(rendered, sig):
    for cond in ("noise_severe", "perspective_mild", "occlusion_severe", "ink_overlay_severe"):
        seed = degrade.condition_seed(sig.record_id, cond)
        a, ra = degrade.apply_condition(rendered, cond, seed)
        b, rb = degrade.apply_condition(rendered, cond, seed)
        assert np.array_equal(a, b)
        assert ra == rb and ra["condition"] == cond and "seed" in ra
    a, _ = degrade.apply_condition(rendered, "noise_mild", 1)
    b, _ = degrade.apply_condition(rendered, "noise_mild", 2)
    assert not np.array_equal(a, b)


def test_unknown_condition_raises():
    with pytest.raises(KeyError):
        degrade.condition_render_config(render.RenderConfig(), "does_not_exist")


def test_lowres_changes_size(rendered):
    img, rec = degrade.apply_condition(rendered, "lowres_severe", 0)
    assert img.shape[1] < rendered.image.shape[1] * 0.5
