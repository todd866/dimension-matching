# Simulated Geometry

**Why Information Alone Cannot Be Alive**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Overview

Life instantiates geometry; AI simulates geometry using information.

*Information* is what can be copied without loss—Shannon entropy, bits, facts. *Geometry* is the constrained manifold within which a system operates—the dimensionality of knowing, not the quantity of known. A database has information; a living cell has geometry.

Large language models simulate geometry using statistical patterns that approximate relational structure, but do not instantiate it: there is no underlying manifold whose dynamics are being maintained. Living systems instantiate geometry through bidirectional cross-scale coupling—molecular dynamics constrain cellular behavior, and cellular behavior maintains molecular organization.

This distinction is measurable. Systems with genuine geometry exhibit **estimator agreement**: independent complexity measures yield consistent answers. Systems that simulate geometry show **estimator mismatch**.

## Key Equation

Match error ε quantifies estimator agreement:

```
ε = |D_geom - D_spec| / ((D_geom + D_spec)/2 + δ)
```

where:
- `D_geom` = geometric proxy (e.g., participation ratio of delay-embedded covariance)
- `D_spec` = spectral proxy (e.g., normalized spectral entropy)
- `δ ≈ 10⁻²` = regularization constant

**Prediction**: Living systems minimize ε; AI systems show systematic mismatch.

## Key Results

- **Information vs Geometry**: Precise distinction with measurable consequences—information is substrate-independent; geometry must be instantiated
- **Match error ε**: Agreement among complexity estimators as signature of genuine geometry
- **Biology as existence proof**: Cross-scale coherence, dormancy as geometry-without-dynamics, death as geometric collapse
- **Why AI simulates**: No intrinsic oscillations, no viability coupling, vertically flat (layers but not levels)
- **Quantum-like structure**: Coarse-graining scale-invariant systems yields non-commuting descriptions

## Testable Predictions

1. **Estimator mismatch in AI**: High D_geom (parameter complexity) but low D_spec (no intrinsic oscillations)
2. **Estimator agreement in biology**: D_geom and D_spec covary across healthy physiological states
3. **ε predicts outcomes**: Match error should predict mortality better than either proxy alone
4. **Temporal ordering**: Rising ε precedes terminal decline (early warning of geometric destabilization)

## Running Simulations

```bash
cd code/
python gmc_simulation.py      # GMC dimension matching demonstration
python generate_figures.py    # Generate manuscript figures
python neural_analysis_v2.py  # Neural network estimator analysis
```

Requirements: numpy, scipy, matplotlib

## Paper

- **Title**: Simulated Geometry: Why Information Alone Cannot Be Alive
- **Target**: BioSystems (hypothesis paper)
- **Status**: In preparation

## Files

```
manuscript.tex       # Main manuscript
manuscript.pdf       # Compiled PDF
cover_letter.tex     # Submission cover letter
references.bib       # Bibliography
code/                # Simulations
  gmc_simulation.py      # GMC dimension matching
  generate_figures.py    # Figure generation
  neural_analysis_v2.py  # Neural network analysis
_archive/            # Previous versions
```

## Citation

```bibtex
@article{todd2025geometry,
  author = {Todd, Ian},
  title = {Simulated Geometry: Why Information Alone Cannot Be Alive},
  journal = {BioSystems},
  year = {2025},
  note = {In preparation}
}
```

## License

MIT License
