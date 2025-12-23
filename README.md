# Dimension Matching in Multiscale Chaotic Systems

**When independently defined complexity measures agree—and when they don't**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

## Overview

In multiscale chaotic systems, geometric complexity (correlation dimension) and spectral complexity (Fourier dimension) can be independently measured. We show these dimensions provably coincide below a critical threshold—a phenomenon we call *dimension matching*—but decouple at a phase transition where coherent structure collapses. Based on the Lin–Qiu–Tan proof of the Garban–Vargas conjecture ([arXiv:2411.13923](https://arxiv.org/abs/2411.13923)), the matching follows a piecewise formula:

$$D_C(\gamma) = D_F(\gamma) = D^*(\gamma) = \begin{cases} 1 - \gamma^2 & \gamma < 1/\sqrt{2} \\ (\sqrt{2} - \gamma)^2 & 1/\sqrt{2} \leq \gamma < \sqrt{2} \end{cases}$$

The critical threshold is **γ = √2 ≈ 1.414**.

## Key Results

- **Dimension matching theorem**: Correlation and Fourier dimensions coincide exactly in subcritical GMC (γ < √2)
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
