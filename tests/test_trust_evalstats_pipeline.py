import inspect
import json

import numpy as np
import pytest

from ecgtrust import analysis, degrade, evalstats, extract, pipeline, render, synth, trust


# ------------------------------------------------------------------ helpers
def _anchors():
    rng = np.random.default_rng(0)
    clean = rng.normal(0, 1, (200, len(trust.FEATURES))) * 0.1 + 0.1
    clean[:, trust.FEATURES.index("f_snr_local")] = 200
    clean[:, trust.FEATURES.index("f_sharp")] = 0.85
    clean[:, trust.FEATURES.index("f_res")] = 8
    clean[:, trust.FEATURES.index("f_snr")] = 200
    clean = np.abs(clean)
    allx = clean.copy()
    allx[100:, :] *= 6
    for n in ("f_snr_local", "f_sharp", "f_res", "f_snr"):
        j = trust.FEATURES.index(n)
        allx[100:, j] = clean[100:, j] / 6
    return trust.fit_anchors(clean, allx)


def _feat(**over):
    f = {n: 0.0 for n in trust.FEATURES}
    f.update(f_snr_local=200.0, f_sharp=0.85, f_res=8.0, f_snr=200.0)
    f.update(over)
    return f


# ------------------------------------------------------------------ trust score
def test_subscore_bounds_and_direction():
    assert trust.subscore(0.0, 0.0, 1.0) == 1.0
    assert trust.subscore(1.0, 0.0, 1.0) == 0.0
    assert trust.subscore(0.5, 0.0, 1.0) == pytest.approx(0.5)
    assert trust.subscore(5, 10, 0) == pytest.approx(0.5)  # higher-is-better direction
    assert trust.subscore(float("nan"), 0, 1) == 0.0
    assert trust.subscore(3.0, 1.0, 1.0) == 1.0  # degenerate anchors must not divide by zero


def test_heuristic_trust_monotone_and_zero_floor():
    a = _anchors()
    good = trust.heuristic_trust(_feat(), a)
    worse = trust.heuristic_trust(_feat(f_unsupported=0.5), a)
    worst = trust.heuristic_trust(_feat(f_unsupported=5.0, f_missing=1.0, f_ambiguity=1.0), a)
    assert 0 <= worst < worse < good <= 100
    assert good > 80


def test_missing_samples_drive_score_down():
    a = _anchors()
    assert trust.heuristic_trust(_feat(f_missing=0.8), a) < trust.heuristic_trust(_feat(f_missing=0.0), a)


def test_classification_rules():
    assert trust.classify(95, True, 80, 50) == "accepted"
    assert trust.classify(65, True, 80, 50) == "uncertain"
    assert trust.classify(30, True, 80, 50) == "rejected"
    # QRS core not observed -> can never be accepted for measurement, however high the score
    assert trust.classify(99, False, 80, 50) == "uncertain"
    assert trust.classify(10, False, 80, 50) == "rejected"


def test_explain_lists_all_groups_worst_first():
    a = _anchors()
    ev = trust.explain(_feat(f_missing=0.9), a)
    assert {e["group"] for e in ev} == set(trust.GROUPS)
    assert ev[0]["group"] == "missing"
    assert all("doc" in f for e in ev for f in e["features"])


def test_trust_features_cannot_see_the_reference():
    # Structural guarantee: the feature functions take only the extraction / beats, never a reference signal.
    for fn in (trust.column_evidence, trust.beat_features, trust.image_level_features, trust.rerender):
        params = set(inspect.signature(fn).parameters)
        assert not (params & {"ref", "x_ref", "reference", "sig", "signal_mv"}), (fn.__name__, params)


def test_learned_model_separates_synthetic_classes():
    rng = np.random.default_rng(1)
    n = 300
    X = np.abs(rng.normal(0, 1, (n, len(trust.FEATURES))))
    y = (X[:, 0] + 0.2 * rng.normal(size=n) > 0.8).astype(int)
    m = trust.fit_learned(X, y)
    s = trust.learned_trust(m, X)
    assert np.corrcoef(s, y)[0, 1] < -0.3  # higher "badness" feature -> lower trust
    assert trust.LearnedModel.from_dict(m.to_dict()).intercept == pytest.approx(m.intercept)


def test_rerender_support_collapses_when_trace_is_corrupted():
    sig = synth.synthetic_signal(seed=3)
    r = render.render_ecg(sig.signal_mv, sig.fs, render.RenderConfig())
    ext = extract.extract_trace(r.image)
    clean_prec = np.nanmean(trust.column_evidence(ext).precision[ext.valid])
    assert clean_prec > 0.5
    ext2 = extract.extract_trace(r.image)
    ext2.y_px = ext2.y_px + 12  # corrupt the extraction: shift the trace 12 px (1.5 mm)
    shifted_prec = np.nanmean(trust.column_evidence(ext2).precision[ext2.valid])
    assert shifted_prec < clean_prec - 0.4


# ------------------------------------------------------------------ evalstats
def test_risk_coverage_toy_example():
    score = np.array([0.9, 0.8, 0.7, 0.6])
    err = np.array([0.0, 0.0, 0.5, 1.0])
    bad = np.array([0, 0, 1, 1.0])
    c = evalstats.risk_coverage(score, err, bad)
    assert c["coverage"].tolist() == [0.25, 0.5, 0.75, 1.0]
    assert c["mean_err"].tolist() == pytest.approx([0.0, 0.0, 0.5 / 3, 0.375])
    assert c["bad_rate"].tolist() == pytest.approx([0, 0, 1 / 3, 0.5])


def test_aurc_perfect_vs_random_vs_inverse():
    rng = np.random.default_rng(0)
    err = rng.random(500)
    perfect = evalstats.aurc(-err, err)
    inverse = evalstats.aurc(err, err)
    flat = evalstats.aurc(np.zeros_like(err), err)  # all tied
    assert perfect < flat < inverse
    assert flat == pytest.approx(err.mean(), rel=1e-6)  # accept-all baseline equals the mean error exactly


def test_choose_thresholds_rule():
    score = np.arange(100, dtype=float)
    bad = (score < 40).astype(float)  # low scores bad
    t = evalstats.choose_thresholds(score, bad, accept_max_bad=0.05, reject_min_bad=0.5)
    # rule: lowest threshold whose accepted set has <=5% bad beats -> 3 bad among 63 accepted -> 37
    assert t["t_accept"] == 37.0 and t["t_reject"] <= t["t_accept"]
    acc = score >= t["t_accept"]
    assert bad[acc].mean() <= 0.05
    # perfectly good data: accept everything
    t2 = evalstats.choose_thresholds(np.arange(50.0), np.zeros(50))
    assert t2["t_accept"] == 0.0


def test_bootstrap_is_deterministic_and_resamples_records():
    rec = np.repeat(np.arange(20), 5)
    err = np.random.default_rng(0).random(100)
    f = {"m": lambda i: float(err[i].mean())}
    a = evalstats.bootstrap_records(rec, f, n_boot=50, seed=3)
    b = evalstats.bootstrap_records(rec, f, n_boot=50, seed=3)
    assert np.array_equal(a["m"], b["m"])
    lo, hi = evalstats.ci95(a["m"])
    assert lo < err.mean() < hi


def test_at_coverage():
    s = np.arange(10.0)
    e = 1.0 - s / 10
    out = evalstats.at_coverage(s, e, (e > 0.5).astype(float), 0.5)
    assert out["coverage"] == 0.5 and out["mean_err"] < e.mean()


# ------------------------------------------------------------------ pipeline
def test_process_record_condition_reproducible_and_complete():
    sig = synth.synthetic_signal(seed=5)
    a = pipeline.process_record_condition(sig, "blur_mild", "test")
    b = pipeline.process_record_condition(sig, "blur_mild", "test")
    for k in ("status", "beats", "alignment", "degradation", "rendering", "calibration"):
        assert a[k] == b[k], k  # bit-for-bit reproducible results (runtime differs, so excluded)
    # provenance requirements
    assert a["source"]["lead"] == "II" and a["source"]["fs_hz"] == 500.0 and a["source"]["units"] == "mV"
    assert a["synthetic_rendering"] is True
    assert "px_per_mm" in a["rendering"] and a["degradation"]["ops"][0]["op"] == "blur"
    assert "thr_frac" in a["extraction_config"]
    assert a["split"] == "test"
    json.dumps(a)  # must be JSON-serialisable
    for beat in a["beats"]:
        assert set(trust.FEATURES) <= set(beat)
        assert beat["bad"] in (True, False)


def test_unreadable_image_gives_unable_row_not_exception():
    sig = synth.synthetic_signal(seed=5)
    # a signal that renders to a trace but whose extraction config forbids everything -> graceful result
    row = pipeline.process_record_condition(sig, "clean", "test", ext_cfg=extract.ExtractionConfig(min_image_px=10**6))
    assert row["status"] == "unable" and row["beats"] == []


def _dummy_model():
    a = _anchors()
    return pipeline.TrustModel(a, None, "heuristic", 60.0, 30.0, {})


def test_analyze_image_end_to_end_and_measurement_gating():
    sig = synth.synthetic_signal(seed=3)
    r = render.render_ecg(sig.signal_mv, sig.fs, render.RenderConfig())
    out = pipeline.analyze_image(r.image, _dummy_model())
    assert out["status"] == "ok" and out["beats"]
    labels = {b["label"] for b in out["beats"]}
    assert labels <= {"accepted", "uncertain", "rejected"}
    for b in out["beats"]:
        m = b["measurements"]
        if b["label"] != "accepted":
            assert m["r_amplitude_mv"] is None and m["rr_ms"] is None and m["status"].startswith("withheld")
        else:
            assert m["r_amplitude_mv"] is not None
    # RR is reported only if both bounding beats are accepted
    for i, b in enumerate(out["beats"]):
        if b["measurements"]["rr_ms"] is not None:
            assert b["label"] == "accepted" and out["beats"][i - 1]["label"] == "accepted"


def test_analyze_image_invalid_inputs():
    m = _dummy_model()
    assert pipeline.analyze_image(None, m)["status"] == "unable"
    assert pipeline.analyze_image(np.zeros((0, 0, 3), np.uint8), m)["status"] == "unable"
    blank = pipeline.analyze_image(np.full((400, 1200, 3), 255, np.uint8), m)
    assert blank["status"] == "unable" and blank["beats"] == []


def test_occluded_beats_are_absent_not_invented():
    """Occlusion removes beats from view. They must disappear from the report (and the valid fraction must
    show the gap), never be filled in or scored as if observed."""
    sig = synth.synthetic_signal(seed=3)
    m = _dummy_model()
    res = {}
    for c in ("clean", "occlusion_severe"):
        cfg = degrade.condition_render_config(render.RenderConfig(), c)
        r = render.render_ecg(sig.signal_mv, sig.fs, cfg)
        img, _ = degrade.apply_condition(r, c, degrade.condition_seed(sig.record_id, c))
        out = pipeline.analyze_image(img, m)
        res[c] = out
        assert out["status"] == "ok"
    assert res["occlusion_severe"]["valid_coverage"] < res["clean"]["valid_coverage"] - 0.05
    assert len(res["occlusion_severe"]["beats"]) < len(res["clean"]["beats"])


def test_blur_lowers_trust_on_synthetic():
    sig = synth.synthetic_signal(seed=3)
    m = _dummy_model()
    t = {}
    for c in ("clean", "blur_severe"):
        cfg = degrade.condition_render_config(render.RenderConfig(), c)
        r = render.render_ecg(sig.signal_mv, sig.fs, cfg)
        img, _ = degrade.apply_condition(r, c, degrade.condition_seed(sig.record_id, c))
        t[c] = np.mean([b["trust"] for b in pipeline.analyze_image(img, m)["beats"]])
    assert t["blur_severe"] < t["clean"] - 20


def test_frozen_model_loads_and_scores(frozen_model_path):
    m = pipeline.TrustModel.load(frozen_model_path)
    assert m.operational in ("heuristic", "learned") and m.t_reject <= m.t_accept
    sig = synth.synthetic_signal(seed=3)
    r = render.render_ecg(sig.signal_mv, sig.fs, render.RenderConfig())
    out = pipeline.analyze_image(r.image, m)
    assert out["status"] == "ok"


def test_rr_never_spans_a_hidden_gap_and_missed_beat_is_not_scored_as_zero_error():
    from ecgtrust import beats as B
    sig = synth.synthetic_signal(seed=3)
    cfg = degrade.condition_render_config(render.RenderConfig(), "occlusion_severe")
    r = render.render_ecg(sig.signal_mv, sig.fs, cfg)
    img, _ = degrade.apply_condition(r, "occlusion_severe", degrade.condition_seed(sig.record_id, "occlusion_severe"))
    ext = extract.extract_trace(img)
    bs = analysis.find_beats(ext)
    ev = analysis.evaluate_record(ext, bs, sig.signal_mv, sig.fs)
    t_ref = np.arange(len(sig.signal_mv)) / sig.fs
    doubled = [b for b in ev["beats"] if b.get("rr_err_ms") is not None and b["rr_err_ms"] > 300]
    assert doubled, "occlusion should hide at least one beat, giving a double-length RR with a large error"
    assert all(not b["rr_eligible"] for b in doubled)  # ... and such an interval must never be eligible for reporting
