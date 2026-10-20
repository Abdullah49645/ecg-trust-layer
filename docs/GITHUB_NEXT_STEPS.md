# GitHub: exact next steps (nothing has been pushed)

Run these yourself from the project folder when you decide to publish. Check `git status` first and make sure no secrets are present (none are used by this project; there are no API keys or `.env` files).

```bash
cd ecg-trust-layer
git init -b main
git add .
git status                 # review: data/ptbxl, logs, demo/_voice, demo/_work and app_runs are git-ignored
git commit -m "ECG Trust Layer: beat-level trust scores for digitized paper ECGs (UnivaBio 2026)"

# create the empty repo on GitHub first (web UI, or:  gh repo create Abdullah49645/ecg-trust-layer --public --source=. --remote=origin )
git remote add origin https://github.com/Abdullah49645/ecg-trust-layer.git
git push -u origin main
```

Notes
- The demo video is about 20 MB: fine for a normal GitHub repo (limit 100 MB per file). For Devpost, upload the video to YouTube/Vimeo and paste the link.
- `results/raw/` (5.7 MB) is committed on purpose: it makes the full-statistics re-derivation work offline.
- The PTB-XL subset used by the full evaluation is **not** committed (`data/ptbxl/` is ignored); `scripts/download_data.py` fetches it.
- After pushing, check that the README images render and that the repository shows the MIT licence.
