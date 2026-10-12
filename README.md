# SIREN-PINN ERT Inversion

A config-driven implementation of a **physics-informed neural network
(PINN)** using **SIREN** (sinusoidal representation networks) for 2D
inversion of direct-current (DC) electrical resistivity tomography (ERT)
field data.

The code inverts Wenner-array apparent-resistivity measurements into a
continuous 2D resistivity model ρ(x, z). The forward operator is the
finite-element ERT solver of [pyGIMLi](https://www.pygimli.org/), which
is used inside the PINN loss to compute the data misfit at every
iteration.

This repository accompanies the manuscript submitted to
**Computers & Geosciences**. It contains one config-driven implementation
of the algorithm. The paper applies the method to four ERT profiles; the
same code reproduces each profile by changing the input file in the YAML
configuration.

---

## Features

- **SIREN network** with `sin(w0 · Wx + b)` activations for a smooth,
  continuously differentiable resistivity model.
- **Physics-based forward operator** — real FEM forward response from
  pyGIMLi's `ERTManager`, evaluated inside the training loop.
- **Data-weighted loss** with depth weighting and 3 % relative error
  model.
- **Smoothness regularisation** on the model gradient, with adaptive
  lambda schedule.
- **Automatic Jacobian caching** — recomputed on a scheduled iteration
  grid to balance accuracy and speed.
- **Dual convergence stopping** — stops when chi-squared / N reaches a
  target OR when the plateau criterion is met.
- **Publication-ready figures** — journal-quality typography, atomic
  save (PNG + PDF).
- **Fully config-driven** — every hyperparameter lives in a single YAML
  file.

---

## Repository layout

siren-pinn-ert-inversion/
- README.md
- LICENSE
- requirements.txt
- .gitignore
- configs/config_profile1.yaml
- data/ERT profile 1 Pangea.xlsx
- scripts/run_profile1.py
- src/__init__.py
- src/run_inversion.py
- results/profile_1/    (created at runtime, git-ignored)

---

## Installation

These instructions are for users who download this repository and want
to run the code on their own machine.

### 1. Create a conda environment

conda create -n pangea python=3.11
conda activate pangea

### 2. Install pyGIMLi

conda install -c conda-forge pygimli

### 3. Install the remaining dependencies

pip install -r requirements.txt

### 4. Verify the installation

python -c "import pygimli, torch, numpy, pandas, yaml; print('OK')"

If this prints OK, the environment is ready.

---

## Data format

The input is an Excel file (*.xlsx) with one sheet named Sheet1
containing one row per four-electrode measurement.

Required columns:
- Ax, Bx — current electrode positions (m)
- Mx, Nx — potential electrode positions (m)
- Apparent Resisitivty — measured apparent resistivity (Ohm.m)

Notes:
- Column matching is case-insensitive. Any column whose name contains
  "resistivity", "resisitivty", or "rho" is accepted.
- Electrode positions may be given as physical coordinates (metres) or
  as integer electrode indices. The code detects which automatically.
- Non-finite or non-positive resistivity values are dropped.

Place the input file inside data/.

---

## Configuration

All hyperparameters are read from configs/config_profile1.yaml.

Key entries:

profile_id: "Profile 1"
input_file: "data/ERT profile 1 Pangea.xlsx"
output_dir: "results/profile_1"

hidden: 64
n_siren_layers: 3
w0: 5.0

n_iters_max: 100
target_chi2: 1.90

learning_rate: 0.005
lambda_reg: 0.1

err_rel: 0.03

To run a different dataset, copy the config, change profile_id,
input_file, and output_dir.

---

## Running the inversion

Via the wrapper script:

python scripts/run_profile1.py

Or directly:

python src/run_inversion.py configs/config_profile1.yaml

Or from Python:

from src.run_inversion import run
results = run("configs/config_profile1.yaml")

A single run takes 5-15 minutes on a standard CPU.

---

## Outputs

All outputs go to results/profile_1/:

- Table1_Profile1.csv — Overall statistics
- Table2_Profile1.csv — Per-Wenner-level statistics
- Table6_Architecture_Profile1.csv — Network architecture
- PINN_Model_Profile1.xlsx — Recovered rho(x,z) and sigma(x,z)
- PINN_DataFit_Profile1.xlsx — Observed vs predicted
- PINN_Results_Profile1.mat — MATLAB bundle
- Fig_Model_Profile1.png — Model figure (raster)
- Fig_Model_Profile1.pdf — Model figure (vector)

The results/ folder is git-ignored.

---

## Reproducing the paper

The paper reports four ERT profiles. Each was run with the same code
and hyperparameters; only the input file changed.

1. Copy the profile's Excel file into data/
2. Copy configs/config_profile1.yaml to config_profile2.yaml
3. Edit three lines:
   profile_id: "Profile 2"
   input_file: "data/ERT profile 2 Pangea.xlsx"
   output_dir: "results/profile_2"
4. Run:
   python src/run_inversion.py configs/config_profile2.yaml

config_profile1.yaml reproduces Profile 1 with R2 = 0.9919 and
chi-squared / N = 1.87.

---

## Citation

If you use this code, please cite the accompanying manuscript:

(Citation will be added upon acceptance.)

---

## License

MIT — see LICENSE.

---

## Contact

Open an issue on GitHub for questions or bug reports.