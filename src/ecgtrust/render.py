"""Deterministic synthetic ECG-paper renderer.

Maps a digital single-lead ECG (millivolts) onto an image of ECG paper: a pink
millimetre grid (minor 1 mm / major 5 mm), a dark trace, and a small text label.
Rendering is a pure function of ``(signal, RenderConfig)``: no randomness here.
Randomness lives in :mod:`ecgtrust.degrade` and is seeded explicitly.

IMPORTANT: these are *synthetic renderings*, not photographs of real paper ECGs.
"""
from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from typing import Optional, Tuple

import cv2
import numpy as np


@dataclass
class RenderConfig:
    # --- paper / physical scales -------------------------------------------------
    mm_per_s: float = 25.0  # horizontal time scale (standard paper speed)
    mm_per_mv: float = 10.0  # vertical amplitude scale (standard gain)
    px_per_mm: float = 8.0  # image resolution (8 px/mm ~ 203 dpi)
    duration_s: float = 10.0
    height_mm: float = 80.0  # plot band height
    margin_mm: float = 6.0  # blank paper left/right/top/bottom of the plot band
    # --- grid ------------------------------------------------------------------
    minor_grid_mm: float = 1.0
    major_grid_mm: float = 5.0
    bg_color: Tuple[int, int, int] = (253, 250, 246)  # RGB
    minor_color: Tuple[int, int, int] = (250, 214, 214)
    major_color: Tuple[int, int, int] = (240, 160, 160)
    grid_darkness: float = 0.0  # 0 = pink grid, 1 = grid drawn as dark gray (grid-interference stress)
    # --- trace -----------------------------------------------------------------
    trace_color: Tuple[int, int, int] = (20, 20, 20)
    trace_contrast: float = 1.0  # 1 = full ink, <1 blends the trace toward the paper colour
    trace_thickness_mm: float = 0.30
    # --- text label --------------------------------------------------------------
    draw_label: bool = True
    label: str = "II"
    # --- vertical placement ------------------------------------------------------
    center_mv: Optional[float] = None  # mV value drawn at the vertical centre; default: mid-range of the signal

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "RenderConfig":
        d = dict(d)
        for k in ("bg_color", "minor_color", "major_color", "trace_color"):
            if k in d and d[k] is not None:
                d[k] = tuple(d[k])
        return cls(**d)


@dataclass
class RenderedECG:
    image: np.ndarray  # uint8 RGB, HxWx3
    config: RenderConfig
    center_mv: float  # mV drawn at vertical centre (the unknown DC offset of the image)
    fs: float
    n_samples: int
    meta: dict = field(default_factory=dict)

    @property
    def size_px(self) -> Tuple[int, int]:
        return self.image.shape[1], self.image.shape[0]

    def x_of_t(self, t_s: np.ndarray) -> np.ndarray:
        c = self.config
        return (c.margin_mm + np.asarray(t_s) * c.mm_per_s) * c.px_per_mm

    def y_of_mv(self, mv: np.ndarray) -> np.ndarray:
        c = self.config
        y_mid = (c.margin_mm + c.height_mm / 2.0) * c.px_per_mm
        return y_mid - (np.asarray(mv) - self.center_mv) * c.mm_per_mv * c.px_per_mm


def _blend(color: Tuple[int, int, int], bg: Tuple[int, int, int], alpha: float) -> Tuple[int, int, int]:
    return tuple(int(round(alpha * c + (1 - alpha) * b)) for c, b in zip(color, bg))  # type: ignore[return-value]


def _draw_grid(img: np.ndarray, cfg: RenderConfig) -> None:
    h, w = img.shape[:2]
    ppm = cfg.px_per_mm
    minor_c, major_c = cfg.minor_color, cfg.major_color
    if cfg.grid_darkness > 0:
        dark_minor, dark_major = (120, 120, 120), (70, 70, 70)
        minor_c = _blend(dark_minor, cfg.minor_color, cfg.grid_darkness)
        major_c = _blend(dark_major, cfg.major_color, cfg.grid_darkness)
    n_minor_x = int(np.floor((w - 1) / (cfg.minor_grid_mm * ppm))) + 1
    n_minor_y = int(np.floor((h - 1) / (cfg.minor_grid_mm * ppm))) + 1
    ratio = int(round(cfg.major_grid_mm / cfg.minor_grid_mm))
    for k in range(n_minor_x):
        x = int(round(k * cfg.minor_grid_mm * ppm))
        if x < w:
            img[:, x] = major_c if k % ratio == 0 else minor_c
    for k in range(n_minor_y):
        y = int(round(k * cfg.minor_grid_mm * ppm))
        if y < h:
            img[y, :] = major_c if k % ratio == 0 else minor_c


def render_ecg(signal_mv: np.ndarray, fs: float, cfg: Optional[RenderConfig] = None) -> RenderedECG:
    """Render ``signal_mv`` (millivolts, sampled at ``fs`` Hz) as an ECG-paper image.

    Only the first ``cfg.duration_s`` seconds are drawn. Raises ``ValueError`` for empty or
    non-finite input.
    """
    cfg = cfg or RenderConfig()
    sig = np.asarray(signal_mv, dtype=np.float64)
    if sig.ndim != 1 or sig.size < 2:
        raise ValueError("signal must be a 1-D array with at least 2 samples")
    if not np.all(np.isfinite(sig)):
        raise ValueError("signal contains non-finite values")
    if fs <= 0:
        raise ValueError("fs must be positive")
    n = min(sig.size, int(round(cfg.duration_s * fs)))
    sig = sig[:n]
    t = np.arange(n) / fs

    ppm = cfg.px_per_mm
    w = int(round((cfg.duration_s * cfg.mm_per_s + 2 * cfg.margin_mm) * ppm))
    h = int(round((cfg.height_mm + 2 * cfg.margin_mm) * ppm))
    img = np.empty((h, w, 3), dtype=np.uint8)
    img[:, :] = cfg.bg_color
    _draw_grid(img, cfg)

    center_mv = float(cfg.center_mv) if cfg.center_mv is not None else float((sig.max() + sig.min()) / 2.0)
    out = RenderedECG(image=img, config=cfg, center_mv=center_mv, fs=fs, n_samples=n)

    # Text label placed inside the plot band's top-left corner (tests text rejection).
    if cfg.draw_label:
        x0 = int((cfg.margin_mm + 1.5) * ppm)
        y0 = int((cfg.margin_mm + 6.0) * ppm)
        cv2.putText(img, f"{cfg.label}  {cfg.mm_per_s:g} mm/s  {cfg.mm_per_mv:g} mm/mV", (x0, y0),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.045 * ppm, (30, 30, 30), max(1, int(round(0.12 * ppm))), cv2.LINE_AA)

    xs = out.x_of_t(t)
    ys = out.y_of_mv(sig)
    SH = 4  # sub-pixel bits for cv2.polylines
    pts = np.stack([xs, ys], axis=1)
    pts = np.round(pts * (1 << SH)).astype(np.int32).reshape(-1, 1, 2)
    thickness_px = max(1, int(round(cfg.trace_thickness_mm * ppm)))
    color = _blend(cfg.trace_color, cfg.bg_color, cfg.trace_contrast)
    cv2.polylines(img, [pts], isClosed=False, color=color, thickness=thickness_px, lineType=cv2.LINE_AA, shift=SH)
    out.meta = {
        "trace_thickness_px": thickness_px,
        "x_first_px": float(xs[0]),
        "x_last_px": float(xs[-1]),
        "n_clipped_samples": int(np.sum((ys < 0) | (ys > h - 1))),
    }
    return out
