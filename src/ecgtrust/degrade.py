"""Controlled, seeded image degradations for the stress-test suite.

A *condition* is a named combination of (a) overrides of the render configuration
(e.g. thicker trace, darker grid) and (b) post-render image operations (blur, noise,
low resolution, perspective, occlusion, ink overlay, reduced contrast). Every random
choice is drawn from a ``numpy`` generator seeded from ``(record_id, condition)`` so a
condition is exactly reproducible. All parameters are returned and stored with results.
"""
from __future__ import annotations

import copy
import zlib
from typing import Dict, List, Tuple

import cv2
import numpy as np

from .render import RenderConfig, RenderedECG

# name -> {"render": {RenderConfig overrides}, "ops": [(op_name, params), ...]}
CONDITIONS: Dict[str, dict] = {
    "clean": {"render": {}, "ops": []},
    "lowres_mild": {"render": {}, "ops": [("lowres", {"scale": 0.6})]},
    "lowres_severe": {"render": {}, "ops": [("lowres", {"scale": 0.35})]},
    "blur_mild": {"render": {}, "ops": [("blur", {"sigma_px": 1.5})]},
    "blur_severe": {"render": {}, "ops": [("blur", {"sigma_px": 3.5})]},
    "noise_mild": {"render": {}, "ops": [("noise", {"sigma": 12.0})]},
    "noise_severe": {"render": {}, "ops": [("noise", {"sigma": 35.0})]},
    "contrast_mild": {"render": {"trace_contrast": 0.65}, "ops": [("contrast", {"alpha": 0.7})]},
    "contrast_severe": {"render": {"trace_contrast": 0.5}, "ops": [("contrast", {"alpha": 0.35})]},
    "perspective_mild": {"render": {}, "ops": [("perspective", {"jitter_frac": 0.012, "rot_deg": 0.8})]},
    "perspective_severe": {"render": {}, "ops": [("perspective", {"jitter_frac": 0.035, "rot_deg": 2.5})]},
    "grid_mild": {"render": {"grid_darkness": 0.5}, "ops": []},
    "grid_severe": {"render": {"grid_darkness": 0.95}, "ops": []},
    "occlusion_mild": {"render": {}, "ops": [("occlusion", {"n_patches": 1, "width_s": (0.2, 0.4)})]},
    "occlusion_severe": {"render": {}, "ops": [("occlusion", {"n_patches": 3, "width_s": (0.3, 0.7)})]},
    "ink_overlay_mild": {"render": {}, "ops": [("ink_overlay", {"n_strokes": 1, "span_s": (0.8, 1.6)})]},
    "ink_overlay_severe": {"render": {}, "ops": [("ink_overlay", {"n_strokes": 3, "span_s": (1.0, 2.4)})]},
    "thin_mild": {"render": {"trace_thickness_mm": 0.15}, "ops": []},
    "thin_severe": {"render": {"trace_thickness_mm": 0.08, "trace_contrast": 0.8}, "ops": []},
    "thick_mild": {"render": {"trace_thickness_mm": 0.5}, "ops": []},
    "thick_severe": {"render": {"trace_thickness_mm": 1.0}, "ops": []},
}

CONDITION_FAMILIES = {
    "clean": "clean",
    **{k: k.rsplit("_", 1)[0] for k in CONDITIONS if k != "clean"},
}


def condition_seed(record_id: str, condition: str, base_seed: int = 7) -> int:
    return (zlib.crc32(f"{record_id}|{condition}".encode()) ^ (base_seed * 2654435761)) & 0xFFFFFFFF


def _bg_color(cfg: RenderConfig) -> Tuple[int, int, int]:
    return cfg.bg_color


def op_lowres(img, rng, cfg, scale):
    h, w = img.shape[:2]
    nw, nh = max(8, int(round(w * scale))), max(8, int(round(h * scale)))
    out = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)
    return out, {"scale": scale, "new_size_px": [nw, nh]}


def op_blur(img, rng, cfg, sigma_px):
    return cv2.GaussianBlur(img, (0, 0), sigma_px), {"sigma_px": sigma_px}


def op_noise(img, rng, cfg, sigma):
    noise = rng.normal(0.0, sigma, img.shape)
    out = np.clip(img.astype(np.float64) + noise, 0, 255).astype(np.uint8)
    return out, {"sigma": sigma}


def op_contrast(img, rng, cfg, alpha):
    mean = np.array(_bg_color(cfg), dtype=np.float64)
    out = mean + (img.astype(np.float64) - mean) * alpha
    return np.clip(out, 0, 255).astype(np.uint8), {"alpha": alpha}


def op_perspective(img, rng, cfg, jitter_frac, rot_deg):
    h, w = img.shape[:2]
    src = np.float32([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]])
    jit = rng.uniform(-jitter_frac, jitter_frac, (4, 2)) * np.array([w, h])
    dst = src + jit.astype(np.float32)
    ang = np.deg2rad(rng.uniform(-rot_deg, rot_deg))
    c, s = np.cos(ang), np.sin(ang)
    ctr = np.array([w / 2.0, h / 2.0], dtype=np.float32)
    rot = np.array([[c, -s], [s, c]], dtype=np.float32)
    dst = ((dst - ctr) @ rot.T + ctr).astype(np.float32)
    M = cv2.getPerspectiveTransform(src, dst)
    out = cv2.warpPerspective(img, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    return out, {"jitter_frac": jitter_frac, "rot_deg_applied": float(np.rad2deg(ang)), "dst_corners_px": dst.round(1).tolist()}


def op_occlusion(img, rng, cfg, n_patches, width_s):
    """Opaque 'tape' patches (paper colour, grid hidden) over parts of the plot band."""
    h, w = img.shape[:2]
    out = img.copy()
    ppm_x = cfg.px_per_mm * cfg.mm_per_s  # px per second (before any lowres; rescaled below)
    scale = w / (cfg.duration_s * cfg.mm_per_s * cfg.px_per_mm + 2 * cfg.margin_mm * cfg.px_per_mm)
    patches = []
    for _ in range(n_patches):
        pw = rng.uniform(*width_s) * ppm_x * scale
        x0 = rng.uniform(cfg.margin_mm * cfg.px_per_mm * scale + 20, w - cfg.margin_mm * cfg.px_per_mm * scale - pw - 20)
        ph = rng.uniform(0.35, 1.0) * (cfg.height_mm * cfg.px_per_mm * scale)
        y_top = cfg.margin_mm * cfg.px_per_mm * scale
        y0 = rng.uniform(y_top, y_top + cfg.height_mm * cfg.px_per_mm * scale - ph)
        x0i, x1i, y0i, y1i = int(x0), int(x0 + pw), int(y0), int(y0 + ph)
        out[y0i:y1i, x0i:x1i] = np.array(cfg.bg_color, dtype=np.uint8)
        patches.append([x0i, y0i, x1i, y1i])
    return out, {"patches_xyxy_px": patches}


def op_ink_overlay(img, rng, cfg, n_strokes, span_s):
    """Dark pen strokes with the same ink as the trace.

    Strokes alternate between (a) *near-parallel* to the trace baseline, 0.25-1.2 mm away (touching or
    merging with the trace around QRS complexes, so the trace becomes ambiguous) and (b) *crossing* the
    baseline at a shallow angle. The baseline row is estimated from the image itself (row with most ink).
    """
    h, w = img.shape[:2]
    out = img.copy()
    scale = w / (cfg.duration_s * cfg.mm_per_s * cfg.px_per_mm + 2 * cfg.margin_mm * cfg.px_per_mm)
    ppm = cfg.px_per_mm * scale
    ink = 255.0 - img.max(axis=2).astype(np.float64)
    rows = np.convolve(ink.sum(axis=1), np.ones(5) / 5.0, mode="same")
    y_base = float(np.argmax(rows))
    strokes = []
    for s_i in range(n_strokes):
        span_px = rng.uniform(*span_s) * cfg.mm_per_s * ppm
        x0 = rng.uniform(cfg.margin_mm * ppm + 10, max(cfg.margin_mm * ppm + 11, w - cfg.margin_mm * ppm - span_px - 10))
        xs = np.linspace(x0, x0 + span_px, 80)
        if s_i % 2 == 0:  # near-parallel
            off = rng.choice([-1.0, 1.0]) * rng.uniform(0.25, 1.2) * ppm
            slope = rng.uniform(-0.02, 0.02)
            ys = y_base + off + slope * (xs - x0) + 0.25 * ppm * np.sin((xs - x0) / (4 * ppm))
            kind = "parallel"
        else:  # crossing
            y0 = y_base + rng.uniform(-3.0, 3.0) * ppm
            slope = rng.choice([-1.0, 1.0]) * rng.uniform(0.08, 0.25)
            ys = y0 + slope * (xs - x0) + 0.5 * ppm * np.sin((xs - x0) / (6 * ppm))
            kind = "crossing"
        pts = np.round(np.stack([xs, ys], 1) * 16).astype(np.int32).reshape(-1, 1, 2)
        thick = max(1, int(round(cfg.trace_thickness_mm * ppm)))
        cv2.polylines(out, [pts], False, (25, 25, 25), thick, cv2.LINE_AA, shift=4)
        strokes.append({"kind": kind, "x0_px": float(x0), "span_px": float(span_px), "slope": float(slope)})
    return out, {"baseline_row_px": y_base, "strokes": strokes}


OPS = {
    "lowres": op_lowres,
    "blur": op_blur,
    "noise": op_noise,
    "contrast": op_contrast,
    "perspective": op_perspective,
    "occlusion": op_occlusion,
    "ink_overlay": op_ink_overlay,
}


def condition_render_config(base: RenderConfig, condition: str) -> RenderConfig:
    if condition not in CONDITIONS:
        raise KeyError(f"unknown condition {condition!r}; known: {sorted(CONDITIONS)}")
    cfg = copy.deepcopy(base)
    for k, v in CONDITIONS[condition]["render"].items():
        setattr(cfg, k, v)
    return cfg


def apply_condition(rendered: RenderedECG, condition: str, seed: int) -> Tuple[np.ndarray, dict]:
    """Apply the post-render operations of ``condition``. Returns ``(image, parameter_record)``."""
    spec = CONDITIONS[condition]
    rng = np.random.default_rng(seed)
    img = rendered.image
    record: List[dict] = []
    for name, params in spec["ops"]:
        img, used = OPS[name](img, rng, rendered.config, **params)
        record.append({"op": name, "params": params, "realised": used})
    return img, {"condition": condition, "seed": int(seed), "render_overrides": spec["render"], "ops": record}
