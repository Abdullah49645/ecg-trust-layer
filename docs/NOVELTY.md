# Related work and novelty assessment

This is a limited (non-exhaustive) literature check done during the build. It is a judgement, not a systematic review.

## What already exists (we do not claim these)

- **Digitization of paper ECGs**, including validated algorithms and AI-based methods: e.g. [Development and Validation of an Algorithm for the Digitization of ECG Paper Images](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC9572306/), [High Precision ECG Digitization Using Artificial Intelligence](https://www.medrxiv.org/content/10.1101/2024.08.31.24312876.full.pdf), and the 2024 PhysioNet Challenge ([Reyna et al.](https://moody-challenge.physionet.org/2024/papers/cinc_paper.pdf)). We did not invent a digitizer; ours is a classical, transparent extractor.
- **Synthetic ECG images:** [ECG-Image-Kit](https://arxiv.org/abs/2307.01946) (Shivashankara et al.) renders signals to realistic, distorted ECG images. The 2024 Challenge also released a database of real ECG images. Our renderer is much simpler.
- **Reference-based evaluation of digitization:** the 2024 Challenge scored digitized signals by signal-to-noise ratio against the reference. That is the right way to evaluate, but needs a reference.
- **Abstention / confidence / selective prediction and risk-coverage evaluation:** [Geifman and El-Yaniv, 2017](https://arxiv.org/abs/1705.08500); pitfalls of AURC in [Overcoming Common Flaws in the Evaluation of Selective Classification Systems](https://arxiv.org/abs/2407.01032). We did not invent confidence scores or abstention.

## What we think is different (and how much)

In our limited search, including the 2024 Challenge overview (which proposes no per-beat or per-image confidence estimate for digitized signals), we did not find a digitization study with all of:

1. a **reference-free**, **beat-level** trust score,
2. computed from evidence in the **image** itself (re-rendering the extracted trace onto the picture, ambiguity, continuity, missing columns, grid calibration, image quality),
3. an explicit **missing** state that is never filled, and
4. **measurement gating** (RR, heart rate, R amplitude only for verifiable beats) evaluated with a pre-specified protocol that includes negative findings.

**Verdict:** the novelty is the *combination and the evaluation protocol*, not any single ingredient. It is an incremental, applied contribution. Its value depends on validating on real images, which we have not done. The measured gain over a simple continuity baseline is modest (test improvement 0.0079, CI [0.0015, 0.0139]; not supported on validation), so we do not claim a breakthrough.
