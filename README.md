# M-TEER Information-Guided Model

An information-guided mathematical modelling framework for intraprocedural M-TEER view selection, simulation, sensitivity analysis, and reproducible computational research.

## Overview

This repository contains the computational framework, model-generated data, figures, tables, and reproducibility materials for an information-guided view-selection model for intraprocedural mitral transcatheter edge-to-edge repair (M-TEER).

The framework represents sequential echocardiographic view selection as an information-allocation problem under uncertainty. It evaluates phase-specific view utility, covariance reduction, information gain, acquisition cost, policy behavior, model comparison, and sensitivity to structural assumptions.

The study is a literature-parameterized mathematical and computational proof-of-concept. It does not use patient-level clinical data and should not be interpreted as a validated clinical decision-support system.

## Repository Structure

- `code/` — Python scripts for model simulation, figure generation, and manuscript generation.
- `data/` — Model-generated outputs, parameter tables, sensitivity analyses, provenance data, and analysis summaries.
- `figures/` — Final figures and table visualizations generated from the computational analyses.
- `LICENSE` — MIT License.
- `.gitignore` — Git exclusion rules.

## Main Scripts

- `code/mteer_view_selection_model.py` — Runs the core information-guided M-TEER model, simulations, policy comparisons, covariance analyses, and sensitivity analyses.
- `code/make_figures.py` — Reproduces the figures and table visualizations from the files in `data/`.
- `code/build_manuscript.py` — Generates manuscript and supplementary documentation from the computational outputs.

## Reproducibility

The computational workflow is organized into three main steps.

### 1. Run the core model

`python code/mteer_view_selection_model.py`

This generates the model outputs and analysis files in the `data/` directory.

### 2. Generate figures and table visualizations

`python code/make_figures.py`

This reads the computational outputs from `data/` and generates the final visualizations in `figures/`.

### 3. Generate the manuscript and companion documentation

`python code/build_manuscript.py`

This uses the analysis summary, literature parameter sources, and generated figures to produce the manuscript and supporting documentation.

The scripts use repository-relative paths, allowing the workflow to be run after cloning or downloading the repository without environment-specific `/mnt/data/` paths.

## Data and Evidence Status

This repository contains literature-parameterized and synthetic computational analyses. No patient-level clinical dataset is included or analyzed.

Model parameters and evidence anchors are documented in `data/Model_Parameters.csv` and `data/Literature_Parameter_Sources.csv`. Simulation outputs, policy analyses, covariance analyses, and sensitivity results are provided in the `data/` directory to support computational transparency and reproducibility.

Numerical outputs should be interpreted as results of the specified mathematical and computational framework rather than estimates of clinical effectiveness or prospective patient outcomes.

## Outputs

The repository provides seven main figures and five table visualizations summarizing the computational framework and its results.

### Figures

1. Model architecture
2. Reference imaging sequence
3. Phase-specific view utility
4. Model comparison using AIC
5. Global sensitivity analysis using PRCC
6. View-selection sensitivity surface
7. Covariance-cycle stability

### Tables

1. Literature anchors
2. Model parameters
3. Model comparison
4. Policy summary
5. Structural sensitivity scenarios

## Requirements

The computational workflow requires Python 3 and the following core packages:

- `numpy`
- `pandas`
- `matplotlib`
- `scipy`
- `python-docx`

Package requirements may vary slightly depending on the script being executed.

## License

This repository is distributed under the MIT License. See `LICENSE` for details.

## Disclaimer

This repository is intended for research and methodological development. The framework is a computational proof-of-concept and has not been prospectively validated for clinical decision-making. It should not be used as a substitute for clinical judgment, institutional protocols, or validated medical guidance.
