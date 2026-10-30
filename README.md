# ECG Signal Processing and QRS Detection

A Python digital signal processing project that examines power-line interference removal and heartbeat detection on ECG data.

## What it does

- Loads an ECG record and expert beat annotations from the MIT-BIH Arrhythmia Database using WFDB.
- Compares a fixed 60 Hz notch filter with an adaptive FFT-based notch filter.
- Applies a Pan–Tompkins-style processing chain to detect QRS complexes.
- Reports residual 60 Hz magnitude, SNR, QRS detection accuracy, missed beats, and false detections against the expert annotations.
- Plots the signal in time and frequency domains and visualizes detector stages.

## Requirements

Python 3. Install the required packages with:

```bash
python -m pip install wfdb numpy scipy matplotlib
```

## Run

```bash
python main.py
```

The script retrieves the selected record and annotations through WFDB, prints comparison metrics, and opens the analysis plots. An internet connection is required for the initial dataset access.

## Project files

- `main.py` — data loading, filtering, QRS detection, metrics, and plots.
- [Paper.pdf](Paper.pdf) — accompanying project report.

## Notes

This is an educational signal-processing project. Detection metrics depend on the selected ECG record and filter settings; run the script to generate the values for the current configuration.
