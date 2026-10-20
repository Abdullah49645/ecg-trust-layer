"""PTB-XL loading, deterministic record selection and download helpers.

PTB-XL (Wagner et al., 2020; PhysioNet, CC BY 4.0) contains *digital* 12-lead
ECGs. It does NOT contain photographs of paper ECGs. This project renders the
digital signals onto synthetic ECG paper to obtain paired (image, reference)
data. See README "Limitations".
"""
from __future__ import annotations

import csv
import dataclasses
import json
import os
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

PTBXL_BASE_URL = "https://physionet.org/files/ptb-xl/1.0.3/"
DATASET_ID = "PTB-XL v1.0.3 (PhysioNet, doi:10.13026/kfzx-aw45)"

# Official PTB-XL recommendation: folds 1-8 train, 9 validation, 10 test.
# strat_fold keeps all records of one patient inside one fold -> no patient leakage across splits.
SPLIT_FOLDS = {"tune": list(range(1, 9)), "val": [9], "test": [10]}


@dataclass
class LoadedSignal:
    """One ECG lead in physical units, exactly as stored in the source record."""

    signal_mv: np.ndarray  # float64, millivolts
    fs: float  # Hz
    lead: str
    units: str
    record_id: str
    dataset: str = DATASET_ID
    patient_id: Optional[str] = None
    extra: Dict[str, object] = field(default_factory=dict)

    @property
    def duration_s(self) -> float:
        return len(self.signal_mv) / self.fs

    def provenance(self) -> dict:
        return {
            "dataset": self.dataset,
            "record_id": self.record_id,
            "patient_id": self.patient_id,
            "lead": self.lead,
            "fs_hz": self.fs,
            "units": self.units,
            "n_samples": int(len(self.signal_mv)),
            **self.extra,
        }


def load_metadata(root: os.PathLike) -> List[dict]:
    """Read ``ptbxl_database.csv`` into a list of dicts (no pandas dependency)."""
    path = Path(root) / "ptbxl_database.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run `python scripts/download_data.py` first (see README)."
        )
    with open(path, newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def select_records(meta: List[dict], n_per_split: Dict[str, int], seed: int = 2026) -> Dict[str, List[dict]]:
    """Deterministically pick records: one per patient, drawn inside the official folds.

    Returns ``{split: [meta_row, ...]}``. Rows keep ``ecg_id``, ``patient_id``,
    ``filename_hr`` (500 Hz) and ``strat_fold``.
    """
    rng = np.random.default_rng(seed)
    out: Dict[str, List[dict]] = {}
    for split, n in n_per_split.items():
        folds = set(SPLIT_FOLDS[split])
        pool = [r for r in meta if int(float(r["strat_fold"])) in folds]
        pool.sort(key=lambda r: int(r["ecg_id"]))
        order = rng.permutation(len(pool))
        seen_patients = set()
        chosen: List[dict] = []
        for idx in order:
            row = pool[int(idx)]
            pid = row["patient_id"]
            if pid in seen_patients:
                continue
            seen_patients.add(pid)
            chosen.append(row)
            if len(chosen) >= n:
                break
        out[split] = chosen
    return out


def record_path(root: os.PathLike, filename_hr: str) -> Path:
    return Path(root) / filename_hr


def load_record(root: os.PathLike, filename_hr: str, lead: str = "II", meta_row: Optional[dict] = None) -> LoadedSignal:
    """Load one lead from a PTB-XL WFDB record in millivolts (physical units)."""
    import wfdb  # imported lazily so unit tests that do not need it still run

    base = str(record_path(root, filename_hr))
    if not (Path(base + ".dat").exists() and Path(base + ".hea").exists()):
        raise FileNotFoundError(f"Record files for {filename_hr} not found under {root}")
    rec = wfdb.rdrecord(base)
    names = [n.upper() for n in rec.sig_name]
    if lead.upper() not in names:
        raise ValueError(f"Lead {lead!r} not in record leads {rec.sig_name}")
    i = names.index(lead.upper())
    sig = np.asarray(rec.p_signal[:, i], dtype=np.float64)
    if not np.all(np.isfinite(sig)):
        sig = np.nan_to_num(sig, nan=0.0)
    extra = {}
    pid = None
    if meta_row is not None:
        pid = meta_row.get("patient_id")
        extra = {"ecg_id": meta_row.get("ecg_id"), "strat_fold": meta_row.get("strat_fold"), "filename_hr": filename_hr}
    return LoadedSignal(
        signal_mv=sig,
        fs=float(rec.fs),
        lead=rec.sig_name[i],
        units=str(rec.units[i]),
        record_id=Path(filename_hr).name,
        patient_id=pid,
        extra=extra,
    )


def _fetch(url: str, dest: Path, retries: int = 6) -> None:
    """Download ``url`` to ``dest`` atomically, with exponential backoff (PhysioNet may throttle)."""
    import time

    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 0:
        return
    last = None
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(url, timeout=60) as resp:
                payload = resp.read()
            tmp = dest.with_suffix(dest.suffix + ".part")
            tmp.write_bytes(payload)
            tmp.replace(dest)
            return
        except Exception as exc:  # noqa: BLE001
            last = exc
            time.sleep(min(30.0, 1.5 * (2**attempt)))
    raise RuntimeError(f"Download failed for {url}: {last}")


def download_metadata(root: os.PathLike) -> None:
    for name in ("ptbxl_database.csv", "scp_statements.csv", "LICENSE.txt"):
        _fetch(PTBXL_BASE_URL + name, Path(root) / name)


def download_record(root: os.PathLike, filename_hr: str) -> None:
    for ext in (".hea", ".dat"):
        _fetch(PTBXL_BASE_URL + filename_hr + ext, Path(root) / (filename_hr + ext))


def save_selection(path: os.PathLike, selection: Dict[str, List[dict]], seed: int) -> None:
    keep = ("ecg_id", "patient_id", "filename_hr", "strat_fold")
    payload = {
        "dataset": DATASET_ID,
        "seed": seed,
        "split_policy": "official PTB-XL strat_fold; tune=folds1-8, val=fold9, test=fold10; one record per patient",
        "splits": {s: [{k: r[k] for k in keep} for r in rows] for s, rows in selection.items()},
    }
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)


def load_selection(path: os.PathLike) -> Dict[str, List[dict]]:
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)["splits"]
