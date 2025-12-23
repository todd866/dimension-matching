# Dimension Matching in Multiscale Chaotic Systems

**When independently defined complexity measures agree—and when they don't**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

## Overview

In multiscale chaotic systems, geometric complexity (correlation dimension) and spectral complexity (harmonic dimension) can be independently measured. We show these dimensions provably coincide below a critical threshold—a phenomenon we call *dimension matching*—but decouple at a phase transition where coherent structure collapses. The matching reflects a martingale-like balance across scales:

$$D_C(\gamma) = D_H(\gamma) = 1 - \frac{\gamma^2}{4}$$

## Key Results

- **Dimension matching theorem**: Correlation and harmonic dimensions coincide exactly in subcritical multiscale chaos
- **Phase transition diagnostic**: Dimension decoupling signals approach to criticality
- **Coherence interpretation**: Matching reflects cross-scale constraint consistency (not phase-locking)
- **Game-theoretic framing**: Subcritical regime = incentive-compatible multiscale contract

## Running Simulations

```bash
cd code
python gmc_simulation.py        # GMC measure construction and dimension estimation
python generate_figures.py      # Generate all publication figures
python agent_game.py            # Multiscale coordination game simulation
python neural_analysis_v2.py    # Neural EEG state analysis
```

## Paper

**Dimension Matching in Multiscale Chaotic Systems: When Correlations and Spectra Coincide**

Todd, I. (2025). *Chaos: An Interdisciplinary Journal of Nonlinear Science* (in prep).

## Citation

```bibtex
@article{todd2025dimension,
  author  = {Todd, Ian},
  title   = {Dimension Matching in Multiscale Chaotic Systems:
             When Correlations and Spectra Coincide},
  journal = {Chaos},
  year    = {2025},
  note    = {In preparation}
}
```

## License

MIT License
