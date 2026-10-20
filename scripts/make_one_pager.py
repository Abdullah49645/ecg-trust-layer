#!/usr/bin/env python
"""One-page PDF summary, generated from results/results_full.json.  python scripts/make_one_pager.py"""
import json
from pathlib import Path

from PIL import Image as PI
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import mm
from reportlab.platypus import Image, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

ROOT = Path(__file__).resolve().parents[1]
R = json.load(open(ROOT / "results" / "results_full.json")); T = R["test"]; h = T["hypotheses"]
ink = colors.HexColor("#1c2125"); acc = colors.HexColor("#0e5a64"); mut = colors.HexColor("#5d666e")
H1 = ParagraphStyle("h1", fontName="Helvetica-Bold", fontSize=19, textColor=ink, leading=22)
Sub = ParagraphStyle("sub", fontName="Helvetica", fontSize=9.5, textColor=mut, leading=12)
H2 = ParagraphStyle("h2", fontName="Helvetica-Bold", fontSize=10.5, textColor=acc, leading=13, spaceBefore=6, spaceAfter=2)
B = ParagraphStyle("b", fontName="Helvetica", fontSize=8.8, textColor=ink, leading=11.5)
doc = SimpleDocTemplate(str(ROOT / "docs" / "ECG_Trust_Layer_one_pager.pdf"), pagesize=A4, leftMargin=15 * mm, rightMargin=15 * mm, topMargin=12 * mm, bottomMargin=10 * mm,
                        title="ECG Trust Layer: one-page summary", author="Abdullah (Abdullah49645)")
f = lambda x: f"{x:.4f}"  # noqa: E731
H1r, H2r = h["H1_trust_beats_accept_all"], h["H2_trust_beats_best_baseline"]
st = [Paragraph("ECG Trust Layer", H1),
      Paragraph("Which heartbeats of a digitized paper ECG can you still verify against the image? · Built for UnivaBio 2026 · research prototype, not a medical device", Sub), Spacer(1, 4),
      Paragraph("What it does", H2),
      Paragraph("Digitizers turn a picture of a paper ECG into a signal, and downstream numbers trust it blindly. ECG Trust Layer gives <b>every heartbeat a trust score computed from the image alone</b> (no reference signal) and reports R amplitude, RR interval and heart rate <b>only for verifiable beats</b>. Unreadable columns stay explicit gaps; unreadable images return “unable”. A local web app shows the image, the extracted trace, per-beat decisions, an evidence inspector, and a real “Reproduce the Evaluation” button.", B),
      Paragraph("Result: held-out test, 80 PTB-XL records, 18,140 beats, 21 image conditions", H2)]
rows = [["Quantity", "AURC (mean beat error over coverage; lower is better)"],
        ["Accept every beat", f(h["aurc_accept_all"]["point"])],
        ["Best simple baseline (continuity)", f(h["aurc_best_baseline"]["point"])],
        ["Trust score (learned)", f(h["aurc_trust_operational"]["point"])],
        ["H1: trust beats accept-all", f"SUPPORTED. AURC lower by {f(H1r['delta_point'])} (-20%); 95% CI of the improvement [{f(H1r['delta_ci95'][0])}, {f(H1r['delta_ci95'][1])}]"],
        ["H2: trust beats best baseline", f"SUPPORTED NARROWLY. Lower by {f(H2r['delta_point'])}; CI [{f(H2r['delta_ci95'][0])}, {f(H2r['delta_ci95'][1])}]. Not supported on validation."]]
tb = Table(rows, colWidths=[56 * mm, 124 * mm])
tb.setStyle(TableStyle([("FONT", (0, 0), (-1, -1), "Helvetica", 8.4), ("FONT", (0, 0), (-1, 0), "Helvetica-Bold", 8.4), ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e6eef0")),
                        ("LINEBELOW", (0, 0), (-1, -1), 0.3, colors.HexColor("#c9d2d6")), ("TEXTCOLOR", (0, 0), (-1, -1), ink), ("VALIGN", (0, 0), (-1, -1), "MIDDLE")]))
st += [tb, Spacer(1, 3), Paragraph("At the app's <b>post-hoc</b> operating point, <b>70% of beats are accepted with a 1.6% bad-beat rate (4.1% overall)</b>.", B), Spacer(1, 4)]
im = Image(str(ROOT / "docs/figures/risk_coverage.png"), width=84 * mm, height=84 * mm * 4.2 / 7.2)
p = ROOT / "docs/screenshots/03b_evidence_inspector.png"; w, hh = PI.open(p).size
sc = Image(str(p), width=94 * mm, height=94 * mm * hh / w)
st += [Table([[im, sc]], colWidths=[86 * mm, 96 * mm], style=[("VALIGN", (0, 0), (-1, -1), "TOP")]),
       Paragraph("Left: risk-coverage on the test split. Right: the evidence inspector for a rejected beat (tilted image).", Sub),
       Paragraph("What did not work (reported honestly)", H2),
       Paragraph("• Our <b>pre-registered operating-point rule was vacuous</b> (it accepted every beat); reported unchanged, with one clearly labelled post-hoc replacement.<br/>• No help for severe blur; 15% of accepted beats are still bad under severe thick-trace.<br/>• Withholding costs: 30% of clean beats withheld; 23% of analyzable images have no reportable heart rate.<br/>• Global image SNR is worse than accepting everything.", B),
       Paragraph("Limits you must know", H2),
       Paragraph("<b>All images are synthetic renderings</b> of PTB-XL lead II, not photographs of real paper ECGs; real-world performance is unknown. No clinical validation; not for diagnosis. The score is a rank-ordered evidence index, not a probability. Single lead, 10 s, standard grid, speed and gain assumed.", B),
       Paragraph("Reproduce", H2),
       Paragraph("<font face='Courier'>python scripts/run_evaluation.py --preset quick</font> (about 30 s) · <font face='Courier'>python app/server.py</font> · 66 automated tests · code MIT; data PTB-XL v1.0.3, CC BY 4.0 (Wagner et al., Sci Data 2020). The demo narration is synthetic speech (Piper TTS). github.com/Abdullah49645/ecg-trust-layer", B)]
doc.build(st)
print("wrote docs/ECG_Trust_Layer_one_pager.pdf")
