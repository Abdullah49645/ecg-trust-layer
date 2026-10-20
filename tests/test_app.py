import io
import json
import sys
import time

import cv2
import numpy as np
import pytest

from ecgtrust import pipeline, render, synth

from test_trust_evalstats_pipeline import _anchors  # noqa: E402


@pytest.fixture(scope="module")
def srv(tmp_path_factory):
    sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1] / "app"))
    import server

    d = tmp_path_factory.mktemp("model")
    p = d / "trust_model.json"
    json.dump({"anchors": _anchors(), "learned": None, "operational_score": "heuristic", "thresholds": {"t_accept": 60.0, "t_reject": 30.0}}, open(p, "w"))
    server.MODEL_PATH = p
    server._model["obj"] = None
    server.app.config["TESTING"] = True
    return server


@pytest.fixture()
def client(srv):
    return srv.app.test_client()


def _png(img_rgb):
    ok, buf = cv2.imencode(".png", cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR))
    assert ok
    return io.BytesIO(buf.tobytes())


def test_index_and_overview(client):
    assert client.get("/").status_code == 200
    j = client.get("/api/overview").get_json()
    assert j["model_ready"] is True and "occlusion_severe" in j["conditions"]


def test_upload_valid_synthetic_image(client):
    sig = synth.synthetic_signal(seed=3)
    img = render.render_ecg(sig.signal_mv, sig.fs, render.RenderConfig()).image
    r = client.post("/api/analyze", data={"image": (_png(img), "ecg.png")}, content_type="multipart/form-data")
    assert r.status_code == 200
    j = r.get_json()
    assert j["status"] == "ok" and j["beats"] and j["source"]["kind"] == "upload"
    assert j["evaluation_only"] is None  # uploads never carry reference-based information
    assert all("evaluation_only" not in b for b in j["beats"])
    tok = j["token"]
    assert client.get(f"/api/image/{tok}/original.png").mimetype == "image/png"
    crop = client.get(f"/api/image/{tok}/beat/0.png")
    assert crop.status_code == 200 and crop.mimetype == "image/png"
    assert client.get(f"/api/image/{tok}/beat/999.png").status_code == 404
    assert client.get("/api/image/nope/original.png").status_code == 404


def test_upload_rejects_bad_input(client):
    assert client.post("/api/analyze", data={}, content_type="multipart/form-data").status_code == 400
    assert client.post("/api/analyze", data={"image": (io.BytesIO(b""), "e.png")}, content_type="multipart/form-data").status_code == 400
    assert client.post("/api/analyze", data={"image": (io.BytesIO(b"not an image at all"), "x.png")}, content_type="multipart/form-data").status_code == 400
    tiny = _png(np.zeros((10, 10, 3), np.uint8))
    assert client.post("/api/analyze", data={"image": (tiny, "t.png")}, content_type="multipart/form-data").status_code == 400


def test_blank_image_is_unable_not_error(client):
    blank = _png(np.full((500, 1500, 3), 255, np.uint8))
    r = client.post("/api/analyze", data={"image": (blank, "b.png")}, content_type="multipart/form-data")
    assert r.status_code == 200
    j = r.get_json()
    assert j["status"] == "unable" and j["beats"] == [] and "message" in j


def test_example_endpoint_labels_reference_info_as_evaluation_only(client, quick_data_available):
    if not quick_data_available:
        pytest.skip("bundled records missing")
    r = client.get("/api/example/0/clean/analyze")
    assert r.status_code == 200
    j = r.get_json()
    assert j["status"] == "ok" and j["source"]["kind"] == "example"
    assert "Never an input" in j["evaluation_only"]["note"]
    assert all("evaluation_only" in b for b in j["beats"])
    assert client.get("/api/example/0/not_a_condition/analyze").status_code == 404
    assert client.get("/api/example/999/clean/analyze").status_code == 404


def test_withheld_measurements_for_non_accepted_beats(client, quick_data_available):
    if not quick_data_available:
        pytest.skip("bundled records missing")
    j = client.get("/api/example/0/blur_severe/analyze").get_json()
    for b in j["beats"]:
        if b["label"] != "accepted":
            assert b["measurements"]["r_amplitude_mv"] is None and b["measurements"]["rr_ms"] is None


def test_eval_endpoints_validate_and_report_idle(client, srv):
    assert client.post("/api/eval/start", json={"mode": "bogus"}).status_code == 400
    assert client.get("/api/results/nope").status_code == 404
    st = client.get("/api/eval/status").get_json()
    assert "state" in st


def test_compare_results_detects_difference(srv):
    a = {"test": {"x": 1.0, "y": [1.0, 2.0], "s": "a"}}
    assert srv.compare_results(a, json.loads(json.dumps(a)))["identical"] is True
    b = {"test": {"x": 1.0, "y": [1.0, 2.5], "s": "a"}}
    c = srv.compare_results(a, b)
    assert c["identical"] is False and c["max_abs_difference"] == pytest.approx(0.5)
