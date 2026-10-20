# Notice, licences and attribution

## This project
ECG Trust Layer source code, configs, tests and documentation: MIT License (see `LICENSE`).
Research prototype. **Not a medical device. Not clinically validated. Not for diagnosis or any clinical decision.**

## Dataset: PTB-XL
This project uses a small subset of the PTB-XL electrocardiography dataset, v1.0.3.

- Wagner, P., Strodthoff, N., Bousseljot, R., Kreiseler, D., Lunze, F.I., Samek, W., Schaeffter, T. (2020).
  *PTB-XL, a large publicly available electrocardiography dataset.* Scientific Data 7, 154.
  https://doi.org/10.1038/s41597-020-0495-6
- Wagner, P., Strodthoff, N., Bousseljot, R., Samek, W., Schaeffter, T. (2022). *PTB-XL, a large publicly available electrocardiography dataset* (version 1.0.3). PhysioNet. https://doi.org/10.13026/kfzx-aw45
- Goldberger, A. et al. (2000). PhysioBank, PhysioToolkit, and PhysioNet. Circulation 101(23), e215-e220.
- Licence: Creative Commons Attribution 4.0 International (CC BY 4.0). The licence text ships with the data (`data/quick/LICENSE.txt`).
- Changes made: only lead II, only the first 10 s of the 500 Hz recordings, converted to mV, and **rendered to synthetic images of ECG paper by this project's code**. The images in this project are computer renderings, not photographs of real paper ECGs.
- The 10 bundled records in `data/quick/` are unmodified PTB-XL files. The full evaluation downloads its own 170-record subset with `scripts/download_data.py`.

## Software
Python, NumPy, SciPy, OpenCV, scikit-learn, wfdb, pandas, Flask, pytest (permissive licences). Optional demo tooling: Playwright, Piper TTS, FFmpeg.

## Demo narration
The demo video's voice-over is **synthetic speech produced by Piper TTS** (voice `en_US-lessac-medium`, from the rhasspy/piper-voices collection). It is not a recording of the author's voice.
