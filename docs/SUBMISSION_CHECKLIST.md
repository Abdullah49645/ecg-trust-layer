# Submission checklist (UnivaBio 2026 on Devpost)

Status as of the build. **Nothing has been submitted, pushed or uploaded.** Items marked MANUAL need you.

## Built and checked (DONE)
- [x] Source repository with reproducible pipeline (`src/ecgtrust`, `scripts/`, `configs/`)
- [x] One-command evaluation: `python scripts/run_evaluation.py --preset quick` (about 30 s) and `--preset full`
- [x] 66 automated tests passing (`pytest -q`)
- [x] Interactive local app with a real **Reproduce the Evaluation** button (`python app/server.py`)
- [x] Machine-readable results: `results/results_full.json`, `results/results_quick.json`, `results/trust_model.json`
- [x] README with app screenshots and the hackathon named at the top
- [x] Scientific report `docs/REPORT.md`, limitations `docs/LIMITATIONS.md`, novelty assessment `docs/NOVELTY.md`
- [x] `LICENSE` (MIT), `NOTICE.md` (PTB-XL attribution, CC BY 4.0), `requirements.txt`, `pyproject.toml`
- [x] Demo video recorded from the real app, with synthetic narration: `demo/ecg_trust_layer_demo.mp4` (plus script, audio, timeline, thumbnail)
- [x] Devpost description draft: `docs/DEVPOST_DESCRIPTION.md`
- [x] One-page PDF: `docs/ECG_Trust_Layer_one_pager.pdf`
- [x] ZIP: `ECG_Trust_Layer_UnivaBio_2026.zip` (integrity-checked, fresh-extract tested)
- [x] GitHub commands prepared (not run): `docs/GITHUB_NEXT_STEPS.md`

## MANUAL: your actions
- [ ] Re-read the hackathon rules/deadline on Devpost yourself (this build deliberately did not check them again)
- [ ] Create the Devpost project; paste `docs/DEVPOST_DESCRIPTION.md`
- [ ] Upload `demo/ecg_trust_layer_demo.mp4` to YouTube/Vimeo (or as the platform allows) and add the link. The narration is synthetic: keep the disclosure line
- [ ] Create the GitHub repo and push using `docs/GITHUB_NEXT_STEPS.md`; add the repo link on Devpost
- [ ] Attach `docs/ECG_Trust_Layer_one_pager.pdf` and the ZIP if the form allows (the rules do not require a PDF)
- [ ] Add team members, thumbnail (`demo/thumbnail.png`), and "Built with" tags
- [ ] Final read-through for claims: images are **synthetic**; no clinical validation; the pre-registered operating point was vacuous; H2 is narrow
- [ ] Press submit

## Claims you must not make
- That it works on real paper ECGs or is clinically validated.
- That the narration is your voice (it is Piper TTS).
- That the accept/reject thresholds were pre-registered (they are post-hoc; the pre-registered rule did nothing).
- That the trust score is a probability.
