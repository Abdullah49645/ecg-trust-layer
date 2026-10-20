# ECG Trust Layer: Devpost description (UnivaBio 2026)

**Tagline:** Which heartbeats of a digitized paper ECG can you still verify against the image?

## Inspiration
Paper ECGs are still everywhere, and digitizers turn a photo of one into numbers. But a digitizer's output looks just as confident on a crisp scan as on a tilted, blurred or smudged one, and every downstream measurement inherits that false confidence. We wanted the tool to say *which beats it can actually verify*, and to say so without needing the original signal, which nobody has at deployment.

## What it does
ECG Trust Layer digitizes a single-lead paper-ECG image and gives **every heartbeat a trust score computed from the image alone**. It then reports measurements (R amplitude, RR interval, heart rate) **only for beats that are verifiable**. Beats it cannot support are rejected, and measurements for them are withheld, not estimated. Unreadable columns are shown as explicit gaps and never filled in; unreadable images return "unable". A local web app lets you analyze bundled examples or your own image, inspect why a beat got its score (six evidence groups, every number visible), and press **Reproduce the Evaluation** to re-run the real pipeline with live progress and compare against the shipped results.

## How we built it
- A classical, transparent extractor: per-strip paper-grid calibration, then a Viterbi dynamic program over candidate ink runs with an explicit *missing* state.
- A reference-free trust score from 14 features in six groups: **re-render support** (the extracted trace is drawn back onto the image and compared with the observed ink), ambiguity, continuity, missing data, grid calibration, and image quality. A heuristic and a learned (logistic regression) version are compared; the operational one is chosen on validation.
- Evaluation on **PTB-XL** (lead II, official folds, one record per patient): 60 tune / 30 validation / 80 test records, 21 simulated image conditions, a pre-registered analysis plan, a feasibility gate, baselines, ablations, and a record-level bootstrap.
- Python, NumPy/SciPy/OpenCV, scikit-learn, Flask, vanilla JS; 66 automated tests; one-command evaluation.

## Results (held-out test; 80 records, 18,140 beats)
- Ranking beats by trust lowers the mean beat error (AURC) from **0.0589 to 0.0468 (-20%)**; the 95% interval of the improvement [0.0059, 0.0175] excludes zero.
- It beats the best simple baseline (a continuity check), **but only narrowly** (improvement 0.0079, CI [0.0015, 0.0139]), and that comparison was *not* supported on validation.
- At the operating point used by the app, **70% of beats are accepted with a 1.6% bad-beat rate (4.1% overall)**.

## Honest limitations and negative results
- **All images are synthetic renderings** of PTB-XL signals. No real paper-ECG photographs were used; real-world performance is unknown. No clinical validation. Not a medical device.
- **Our pre-registered operating-point rule was vacuous** (it accepted every beat). We report it unchanged and added one clearly labelled post-hoc operating point.
- The score does not help for severe blur, and 15% of accepted beats are still bad under severe thick-trace conditions.
- Withholding has a cost: 30% of clean beats are withheld, and 23% of analyzable images have no reportable heart rate.
- Full disclosure of protocol deviations is in the report.

## Challenges
Making the extractor robust (grid scale ambiguity under heavy down-sampling, perspective, occlusion) without special-casing; and building an evaluation that cannot flatter us: an RR-definition bug that hid missed beats was found and fixed before any test result existed.

## Accomplishments
A working end-to-end system whose claims are bounded by what we measured, with every figure reproducible from one command and from a button in the app.

## What we learned
Image-level quality metrics (global SNR) are *worse* than accepting everything; beat-level evidence from the image is what carries the signal. And a pre-registered threshold rule can fail for a boring reason (the base rate was below the target), which is worth reporting.

## What's next
Validate on real photographed and scanned ECG images (for example the 2024 PhysioNet Challenge image database), multi-lead layouts, a better image-quality feature group, and calibration of the score into a probability.

## Built with
Python · NumPy · SciPy · OpenCV · scikit-learn · wfdb · Flask · JavaScript · PTB-XL (CC BY 4.0)

## Demo video note
The demo is a screen recording of the real running app. The narration is **synthetic speech (Piper TTS)**, not the author's voice.
