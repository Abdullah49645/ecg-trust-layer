# Demo video

| File | What it is |
|---|---|
| `ecg_trust_layer_demo.mp4` | Final video: screen recording of the **real running app** + synthetic narration (H.264/AAC, 1440×900) |
| `screen_recording.mp4` | The silent screen recording |
| `voiceover.wav` | The narration track (mono, 22.05 kHz), placed at the measured segment offsets |
| `voiceover_script.md` | Exact narration text, with timestamps |
| `demo_timeline.md`, `timeline.json` | Segment timeline |
| `narration.json` | Source of the narration and the UI actions per segment |
| `record_demo.py` | The recorder (Playwright drives the real app; Piper synthesizes; FFmpeg muxes) |
| `thumbnail.png` | A frame from the video |

**The voice is synthetic** (Piper TTS, `en_US-lessac-medium`). It is not the author's voice. A small on-screen badge says so. There are no burned-in subtitles.

## Regenerate

```bash
pip install playwright piper-tts && playwright install chromium     # plus ffmpeg on PATH
mkdir -p demo/_voice && cd demo/_voice
curl -LO https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/lessac/medium/en_US-lessac-medium.onnx
curl -LO https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/lessac/medium/en_US-lessac-medium.onnx.json
cd ../.. && python demo/record_demo.py       # about 6 min; it starts the app itself on port 5070
```

The voice model (63 MB) is not bundled. Edit `narration.json` to change the narration or the UI actions; the recorder re-synthesizes and re-records.
