# Curvature-Subspace Criteria for Local Loss-Landscape Stabilization

[![paper](https://img.shields.io/badge/paper-preprint-red.svg)](paper/main.pdf)
[![code](https://img.shields.io/badge/code-repository-green.svg)](code/)

This is the official companion repository for the paper **Curvature-Subspace Criteria for Local Loss-Landscape Stabilization** by [Nikita Kiselev](https://kisnikser.github.io/) and [Andrey Grabovoy](https://scholar.google.com/citations?user=ZtI9pgsAAAAJ&hl=en&oi=sra).

<div align="center">
    <img alt="Subspace mean-squared criterion (schematic)" width="520" src="paper/figures/criterion_squared_subspace.png">
</div>

<br>

> **Abstract:** Deep neural loss landscapes are highly anisotropic: most local curvature is concentrated in a small number of directions. We study how such landscapes stabilize as the training set grows one sample at a time. Previous work introduced pointwise and full-space mean-squared criteria for local landscape increments, but these criteria do not exploit local curvature structure and may average the signal over many weakly informative directions. We introduce a curvature-aware subspace criterion that probes local landscape change in the leading eigenspace of the empirical Hessian near a trained solution. Under a local quadratic model, we show that this criterion preserves the same mean-squared convergence order as the full-space criterion, while depending on the subspace dimension rather than the ambient parameter dimension. We also derive a spectral interpretation under an additional stable-eigenspace assumption and describe scalable estimators based on Hessian--vector products and subspace Monte Carlo. Experiments on nanoGPT-style transformers confirm that the subspace criterion robustly preserves the mean-squared decay across different subspace dimensions and perturbation scales. However, at the tested perturbation scales, the scalar value gap dominates the criterion, making different subspace choices empirically indistinguishable, and the quadratic proxy estimators overestimate the direct criterion by several orders of magnitude. These results support the proposed criterion as a well-behaved curvature-aware observable while highlighting concrete limitations of the current observational design.

## Repository structure

This repository is structured as follows:

- **`code`** — experiment code and plotting; see [`code/README.md`](code/README.md).
- **`paper`** — preprint [`paper/main.pdf`](paper/main.pdf) and LaTeX sources (`main.tex`, `references.bib`, NeurIPS style).

## Citation

If you find our work helpful, please cite us.

```bibtex
@misc{kiselev2026curvaturesubspace,
    title={Curvature-Subspace Criteria for Local Loss-Landscape Stabilization},
    author={Kiselev, Nikita and Grabovoy, Andrey},
    year={2026},
    note={Manuscript. NeurIPS submission.}
}
```

## Licence

Our project is MIT licensed. See [LICENSE](LICENSE) for details.
