# Demo voice-over script

Narration is **synthetic speech** (Piper TTS, voice `en_US-lessac-medium`). It is not the author's voice. The video shows the real running app; every number spoken here is read from `results/results_full.json`.

**[0:00.5] intro**

Digitized paper ECGs are usually trusted blindly. ECG Trust Layer scores every heartbeat by how well the original image supports it. This is a research prototype, and every image here is a synthetic rendering of a PTB-XL signal, not a photograph of a real paper ECG.

**[0:18.3] clean**

A clean example. The tool reads the grid, extracts the trace, finds each beat, and scores it. The summary counts accepted, uncertain and rejected beats.

**[0:29.0] inspect**

Click a beat to see why. Six kinds of evidence feed the score: re-render support, ambiguity, continuity, missing data, grid calibration and image quality, and every number can be opened. Measurements are reported only for accepted beats.

**[0:45.2] occlusion**

Now part of the trace is hidden. The unreadable columns show as a gap and are never filled in. Beats whose windows touch the gap lose trust, and a beat hidden completely would vanish from the list rather than be invented.

**[0:57.7] hard**

With a strongly tilted image the grid disagrees with itself, so every beat is rejected and all measurements are withheld. The dashed panel is evaluation only. It uses the original signal, which the score never sees, and confirms these beats really were wrong.

**[1:13.8] result**

On the held-out test of eighty records and over eighteen thousand beats, ranking by trust cuts the average error from point zero five eight nine to point zero four six eight, twenty percent lower, with an interval that excludes zero. It also beats the best simple baseline, but only just.

**[1:30.8] chart**

The teal line stays below accept-everything, but it is still far from the dotted oracle, so there is real headroom.

**[1:37.8] op**

At the operating point the app uses, seventy percent of beats are accepted, and only one point six percent of those are bad, against four point one percent overall.

**[1:48.3] prereg**

I have to be honest about this operating point. It was chosen after the pre-registered rule turned out to be vacuous: that rule accepted every beat. That negative result is reported here, unchanged.

**[2:00.3] weak**

It does not work everywhere. With severe thick traces, fifteen percent of accepted beats are still bad, and for severe blur the score does not help. The gate also has a cost: for about a quarter of images, no heart rate could be reported.

**[2:14.9] repro**

Everything here can be reproduced with this button. It runs the real pipeline on this machine: render, degrade, digitize, score, and statistics. The quick preset uses ten held-out records and takes about half a minute.

**[3:15.7] repro_done**

It finished and matches the shipped results exactly. Ten records only prove reproducibility. The confirmatory result is the eighty-record test.

**[3:25.4] limits**

The limits: synthetic images, no clinical validation, an evidence index rather than a probability, and only R amplitude, R R interval and heart rate are measured. This is not a medical device.

**[3:38.2] outro**

ECG Trust Layer: show which beats you can still verify. The narration is synthetic speech from Piper.
