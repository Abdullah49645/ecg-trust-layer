# Limitations and intended use

ECG Trust Layer is a **research prototype** built for UnivaBio 2026. It is **not a medical device**, has **no clinical validation**, and must not be used for diagnosis, triage, treatment or any clinical decision.

## Intended use

- Research on ECG digitization: deciding which reconstructed beats are verifiable enough to include in an analysis, and showing why.
- A transparent quality gate for three measurements only: RR interval, heart rate, and R amplitude above the pre-QRS baseline.

## Not intended for

- Diagnosis, rhythm classification, or any patient-level decision.
- QT/ST/interval analysis, multi-lead layouts, or 12-lead reconstruction. None of these is implemented.
- Real photographs or scans of paper ECGs: **untested** (see below).

## What the evidence does and does not show

| Claim | Status |
|---|---|
| Ranking beats by the trust score lowers beat error on held-out synthetic images | Supported (AURC 0.0589 -> 0.0468, CI of improvement [0.0059, 0.0175]) |
| The score beats the best simple baseline (continuity) | Supported only narrowly on test (CI [0.0015, 0.0139]); **not supported on validation** |
| The accept/uncertain/reject thresholds are meaningful | Only the **post-hoc** operating point is; the **pre-registered** rule was vacuous (accepted everything) |
| It works on real paper ECGs | **Unknown. Not tested.** |
| It is clinically useful or safe | **No claim.** Not evaluated. |

## Known limitations

1. **Synthetic images.** All images are computer renderings of PTB-XL lead II with simulated degradations. The renderer and extractor share a pixel-grid model, which makes clean extraction easier than for real scans. Real-world performance is unknown and probably worse. Validation on real images (for example the ECG-Image-Database released with the 2024 PhysioNet Challenge) is the most important next step.
2. **Degradation mixture is our choice.** Pooled results depend on which conditions are included; use the per-condition table.
3. **Failure modes we measured.** The score does not help under *severe blur* (accepted bad-beat rate 7.7% vs 7.5% overall) and misses many bad beats under *severe thick trace* (15.1% of accepted beats are bad). On *clean* images 30% of beats are withheld although only 0.9% are bad.
4. **Withholding is a cost.** At the post-hoc operating point 23% of analyzable test images have no reportable heart rate.
5. **Evidence index, not probability.** A score of 80 does not mean "80% correct". Scores are rank-ordered.
6. **Visible beats only.** A beat hidden completely (for example by occlusion) vanishes from the list; it is shown as a gap and never invented. Absolute DC level is unobservable; amplitudes are relative to the pre-QRS baseline.
7. **Scope.** Single lead (II), 10 s, standard 25 mm/s and 10 mm/mV assumed, standard pink/red 1 mm/5 mm grid, dark trace on light paper. Inputs outside this are returned as `unable` where detected, but a confidently wrong reading cannot be excluded.
8. **Small evaluation.** 80 test records from one dataset (one record per patient); confidence intervals are wide for the comparison with the baseline.
9. **Development exposure.** See the disclosures in `docs/REPORT.md` Section 7 (a 6-record smoke run on test-fold records; extractor developed against the same simulated degradations used to evaluate it).

## Narration disclosure

The demo video's voice-over is **synthetic speech** (Piper TTS, voice `en_US-lessac-medium`). It is not a recording of the author's voice.
