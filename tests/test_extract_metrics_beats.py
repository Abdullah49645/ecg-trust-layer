import numpy as np
import pytest

from ecgtrust import analysis, beats, degrade, extract, metrics, render, synth


def _extract(sig, cond="clean"):
    cfg = degrade.condition_render_config(render.RenderConfig(), cond)
    r = render.render_ecg(sig.signal_mv, sig.fs, cfg)
    img, _ = degrade.apply_condition(r, cond, degrade.condition_seed(sig.record_id, cond))
    return extract.extract_trace(img), r, img


# ---------------- extraction ----------------
def test_clean_extraction_calibration_and_accuracy(sig):
    ext, r, _ = _extract(sig)
    assert ext.ok
    assert ext.calibration["px_per_mm_x"] == pytest.approx(8.0, abs=0.05)
    assert ext.calibration["px_per_mm_y"] == pytest.approx(8.0, abs=0.05)
    assert ext.t_s[-1] == pytest.approx(10.0, abs=0.05)  # time calibration
    t_ref = np.arange(len(sig.signal_mv)) / sig.fs
    al = metrics.align_and_score(ext.t_s, ext.mv, ext.valid, t_ref, sig.signal_mv)
    assert al["ok"]
    assert al["dc_lag"]["nrmse"] < 0.06
    assert al["dc_lag"]["gain"] == pytest.approx(1.0, abs=0.05)  # amplitude calibration (10 mm/mV)
    assert al["valid_coverage"] > 0.98
    # three-stage reporting is present
    assert set(["raw", "dc", "dc_lag"]) <= set(al)


def test_text_and_grid_not_mistaken_for_trace(sig):
    ext, _, _ = _extract(sig)
    # extraction extent must start at the trace, not at the text label (which lies inside the plot band)
    assert ext.stats["x_extent_px"][0] > 40 and ext.stats["x_extent_px"][1] < 2060


def test_missing_samples_are_masked_not_filled(sig):
    ext, _, img = _extract(sig, "occlusion_severe")
    assert ext.ok
    invalid = ~ext.valid
    assert invalid.any()
    assert np.all(np.isnan(ext.mv[invalid])) and np.all(np.isnan(ext.y_px[invalid]))


def test_dark_grid_and_noise_still_extract(sig):
    for c in ("grid_severe", "noise_severe", "contrast_severe"):
        ext, _, _ = _extract(sig, c)
        assert ext.ok, (c, ext.reason)


def test_blank_image_is_unable_not_a_crash():
    img = np.full((600, 1800, 3), 255, dtype=np.uint8)
    res = extract.extract_trace(img)
    assert res.status == "unable" and res.reason


def test_noise_only_image_is_unable():
    rng = np.random.default_rng(0)
    img = rng.integers(0, 255, (400, 1200, 3), dtype=np.uint8)
    res = extract.extract_trace(img)
    assert res.status == "unable"


@pytest.mark.parametrize("bad", [None, np.zeros((0, 0, 3), np.uint8), np.zeros((4, 4, 3), np.uint8), np.zeros((10,), np.uint8), np.zeros((50, 50, 5), np.uint8)])
def test_invalid_image_raises_extraction_error(bad):
    with pytest.raises(extract.ExtractionError):
        extract.extract_trace(bad)


def test_grayscale_input_accepted(sig):
    _, _, img = _extract(sig)
    gray = img.mean(axis=2).astype(np.uint8)
    res = extract.extract_trace(gray)
    assert res.status in ("ok", "unable")  # must not crash; grid colour is lost so either is acceptable


# ---------------- metrics / alignment ----------------
def test_basic_metrics_known_values():
    r = np.array([0.0, 1.0, 0.0, -1.0, 0.0, 1.0, 0.0, -1.0])
    x = r + 0.1
    m = metrics.basic_metrics(x, r, amp=2.0)
    assert m["rmse_mv"] == pytest.approx(0.1)
    assert m["nrmse"] == pytest.approx(0.05)
    assert m["max_abs_err_mv"] == pytest.approx(0.1)
    assert m["pearson"] == pytest.approx(1.0)  # correlation ignores the offset -> must not be the only metric
    assert m["gain"] == pytest.approx(1.0)


def test_alignment_recovers_known_lag_and_offset(sig):
    t_ref = np.arange(len(sig.signal_mv)) / sig.fs
    t = np.arange(0, 9.5, 0.005)
    true_lag, true_off = 0.012, 0.37
    x = np.interp(t + true_lag, t_ref, sig.signal_mv) + true_off
    al = metrics.align_and_score(t, x, np.ones(t.size, bool), t_ref, sig.signal_mv)
    assert al["lag_s"] == pytest.approx(true_lag, abs=0.001)
    assert al["offset_mv"] == pytest.approx(true_off, abs=0.01)
    assert al["dc_lag"]["rmse_mv"] < 0.01
    assert al["raw"]["rmse_mv"] > al["dc_lag"]["rmse_mv"]


def test_alignment_too_few_valid_samples():
    t = np.arange(10) * 0.01
    out = metrics.align_and_score(t, np.zeros(10), np.zeros(10, bool), t, np.zeros(10))
    assert out["ok"] is False


# ---------------- beats ----------------
@pytest.mark.parametrize("hr", [55, 72, 110])
def test_r_peak_detection_rate(hr):
    s = synth.synthetic_ecg(hr_bpm=hr, seed=1)
    idx, pol = beats.detect_r_peaks(s, 500.0)
    assert pol == 1
    rr = np.diff(idx) / 500.0
    assert 60.0 / np.mean(rr) == pytest.approx(hr, rel=0.03)


def test_inverted_qrs_polarity():
    s = -synth.synthetic_ecg(seed=2)
    idx, pol = beats.detect_r_peaks(s, 500.0)
    assert pol == -1 and idx.size >= 8


def test_r_amplitude_and_matching():
    s = synth.synthetic_ecg(seed=4)
    idx, pol = beats.detect_r_peaks(s, 500.0)
    amp = beats.r_amplitude(s, 500.0, int(idx[2]), pol)
    assert 0.9 < amp < 1.5  # QRS component amplitude 1.1 mV above the pre-QRS baseline (+ small Q)
    m = beats.match_beats(np.array([1.0, 2.0, 3.5]), np.array([1.02, 2.5, 3.46]), tol_s=0.1)
    assert m == [0, None, 2]


def test_short_signal_has_no_beats():
    idx, _ = beats.detect_r_peaks(np.zeros(100), 500.0)
    assert idx.size == 0


def test_find_beats_on_failed_extraction_is_empty():
    res = extract.extract_trace(np.full((300, 900, 3), 250, np.uint8))
    assert analysis.find_beats(res).n == 0


def test_find_beats_matches_reference_on_clean(sig):
    ext, _, _ = _extract(sig)
    bs = analysis.find_beats(ext)
    t_ref = np.arange(len(sig.signal_mv)) / sig.fs
    ev = analysis.evaluate_record(ext, bs, sig.signal_mv, sig.fs)
    assert ev["ok"] and ev["n_matched"] == ev["n_ref_beats"] == bs.n
    assert ev["hr_abs_err_bpm"] < 1.0
    assert all(b["matched"] for b in ev["beats"])
