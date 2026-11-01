# Project 2 — Reproducibility Package

## Research title

Prediction of Packet Error Rate in Vehicular Communication Networks
Using Machine Learning: Performance, Generalization, and
Hardware Complexity Analysis

## Scope

This package preserves the reference models, evaluation results,
statistical analyses, manuscript figures, and analytical hardware
complexity artifacts used in Project 2.

## Reference models

Four model families were evaluated:

- Ridge regression
- Histogram Gradient Boosting
- Multilayer Perceptron
- Convolutional Neural Network

The reference evaluation uses nine folds per model family,
corresponding to 36 reference checkpoints.

Classical checkpoints:
`results/juelich_cv_h10_common40_refit/models/`

Neural checkpoints:
`results/neural_cv_h10/models/`

## Evaluation results

Within-domain and cross-dataset results are located in:

- `results/juelich_cv_h10_common40_refit/`
- `results/neural_cv_h10/`
- `results/cross_dataset_h10_common40/`
- `results/m05_berlin_full/`

## Statistical analysis

Uncertainty estimates and paired comparisons are available in:

- `results/bootstrap_uncertainty_h10_common40/`
- `results/berlin_grouped_uncertainty/`

## Hardware complexity

Analytical model profiles and hardware export artifacts are located in:

- `results/model_profile_h10/`
- `results/hardware_export_h10/`

These are analytical complexity and storage estimates.

They must not be interpreted as FPGA synthesis,
post-route resource utilization, measured latency,
or physical power consumption.

## Manuscript figures

Final figure files are located in:

`results/manuscript_v1/figures/`

## Artifact verification

From the repository root, execute:

    sha256sum -c docs/reproducibility/SHA256SUMS.txt

The manifest contains SHA-256 checksums of the selected archived
artifacts. Successful checksum verification confirms file integrity,
not independent computational reproduction.

## Software environment

See:

`docs/reproducibility/environment_snapshot.txt`

## Reproducibility limitations

The archived artifacts support inspection of reported results and
verification of file integrity.

Independent reproduction of all model training, dataset generation,
cross-dataset evaluations, statistical calculations, and manuscript
figures has not yet been established by this package.

Some experiments may depend on external datasets, local paths,
specific library versions, or previously generated intermediate files.

The availability of a checkpoint does not, by itself, establish
reproducibility of the original training procedure.

## Relationship to Project 3

Project 3 uses the reference prediction models developed in
Project 2 for hardware-aware quantization and FPGA implementation.

Project 3 synthesis and implementation results are separate
from the analytical FPGA complexity assessment in Project 2.
