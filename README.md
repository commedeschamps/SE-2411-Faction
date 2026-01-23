# FACTION: Enigma & Operation Kessler

This repository contains the coursework for Phase 1: **ENIGMA** and **Operation Kessler**. The work focuses on generating noisy signals, identifying underlying empirical laws, reconstructing missing data, and making predictions from fitted models. A full write-up is in `REPORT.pdf`.

## Repository Layout
- `REPORT.pdf` — final report describing methods, models, and results.
- `enigma/src/` — scripts for generating and decrypting the ENIGMA signal.
- `kessler/data/raw/` — raw Alpha/Beta/Gamma sensor CSVs.
- `kessler/src/alpha/alpha.py` — Alpha signal model fitting script.
- `kessler/src/plots/` — saved plots for Alpha/Beta/Gamma.

## Requirements
Python 3 with:
- `numpy`
- `pandas`
- `matplotlib`
- `scipy`

## Quick Start
Run scripts from their respective folders so relative outputs land in the right place.

### ENIGMA (Phase 1)
1) Generate a secret signal (exponential by default) and save a noisy CSV:
```bash
cd enigma/src
python3 generate.py
```
This writes `faction_signal.csv` in the current directory and plots the clean vs. noisy signal.

2) Decrypt the signal by fitting a quadratic model:
```bash
cd enigma/src
python3 decrypt.py
```
This reads `faction_signal.csv`, fits a model, saves `final_quadratic_decryption.png`, and prints fitted coefficients.

### Operation Kessler
Run the Alpha fit script (reads the CSV from the repo path):
```bash
cd kessler/src/alpha
python3 alpha.py
```
The script fits an exponential + sinusoidal model and displays the fitted curve.

Beta/Gamma data files are in:
- `kessler/data/raw/Beta_Drift.csv`
- `kessler/data/raw/Gamma_Thermal.csv`

Optional quick plot for Beta/Gamma:
```bash
python3 - <<'PY'
import pandas as pd
import matplotlib.pyplot as plt

for name in ["Beta_Drift", "Gamma_Thermal"]:
    df = pd.read_csv(f"kessler/data/raw/{name}.csv")
    plt.figure()
    plt.plot(df["Time"], df["Sensor_Reading"])
    plt.title(name)
    plt.xlabel("Time")
    plt.ylabel("Sensor_Reading")

plt.show()
PY
```

Plots for all three signals are stored in `kessler/src/plots/`.

## Report Summary
The report documents:
- **ENIGMA**: generation of a secret exponential law with Gaussian noise and decryption of rival signals via model selection and parameter estimation.
- **Operation Kessler**: PURGE → FIT → RECONSTRUCT → PREDICT workflow for Alpha (altitude), Beta (drift), and Gamma (thermal) signals, including interpolation for blackout intervals and prediction at `t = 100`.

## Report Results (from REPORT.pdf)
### ENIGMA (Phase 1)
- Secret signal law: `y = a * exp(b * x)` with `a = 10.0`, `b = 10.0`, `x in [0, 10]` (20 points).
- Noise model: `epsilon ~ N(0, 5)` and `y_encrypted = y_true + epsilon`.
- Rival signal model (reported reconstruction): `y = 0.2 * x^2.5 + 10`.

### Operation Kessler (FIT + PREDICT)
- **Alpha (Altitude)**:
  - Model form: `y(x) = A * exp(-B * x) + C * sin(D * x) + E`
  - Reported parameters: `A = 9768.700`, `B = 0.016`, `C = 321.412`, `D = 0.498`, `E = 303.732`
  - Prediction: `y(100) = 2124.386`
- **Beta (Drift)**:
  - Model: `y(x) = 0.007 * x^3 - 0.845 * x^2 + 50.169 * x - 32.668`
  - Prediction: `y(100) = 3203.661`
- **Gamma (Thermal)**:
  - Reported final equation: `y(x) = 653.278 * ln(1.224 + 1.219 * x) + 66.398`
  - Prediction: `y(100) = 3210.666`
- Reconstruction step: blackout intervals filled using cubic spline interpolation (no extrapolation).

## Notes
- `enigma/src/decrypt.py` expects `faction_signal.csv` in the working directory.
- `kessler/src/alpha/alpha.py` reads `kessler/data/raw/Alpha_Altitude.csv` via a repo-relative path.
