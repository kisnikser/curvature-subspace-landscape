# Curvature-Aligned Probing for Local Loss-Landscape Stabilization

[![paper](https://img.shields.io/badge/paper-preprint-red.svg)](paper/neurips_2026.pdf)
[![code](https://img.shields.io/badge/code-repo-blue.svg)](code/README.md)

This is the official companion repository for the paper **Curvature-Aligned Probing for Local Loss-Landscape Stabilization** by [Nikita Kiselev](https://kisnikser.github.io/) and [Andrey Grabovoy](https://scholar.google.com/citations?user=ZtI9pgsAAAAJ&hl=en&oi=sra).

<div align="center">
    <img alt="Subspace mean-squared criterion (schematic)" width="520" src="paper-tmlr/figures-old/criterion_squared_subspace.png">
</div>

<br>

> **Abstract:** Local loss-landscape stabilization under sample growth is typically measured either pointwise or through isotropic averaging in the full parameter space. Despite practical value, both choices probe directions that contribute little to the dominant local deformation of strongly anisotropic neural landscapes. We recast stabilization as an observational problem and introduce a unified family of criteria parameterized by an aggregation order and a probing distribution; within this family we propose a curvature-aligned criterion that probes the loss increment field in the top-D eigenspace of the empirical Hessian near a trained solution. Solely from a local quadratic model, we prove that criterion preserves the quadratic mean-squared rate of the full-space criterion while replacing ambient-dimension curvature dependence with dependence on the subspace dimension D; a corollary gives a closed-form spectral expression and a proposition identifies the top-D eigenspace as extremal within the eigenspace-aligned family. We also derive scalable estimators based on Hessian--vector products, subspace Monte Carlo, and a closed-form Gaussian-moment proxy. On a decoder-only transformer, a curvature-aligned probe occupying a tiny fraction of parameter space already reproduces the full-space mean-squared signal to within numerical noise throughout the validated local regime, and the closed-form estimator is orders of magnitude faster than direct Monte Carlo after subspace construction.

## Repository structure

This repository is structured as follows:

- **`code`** — [nanochat](https://github.com/karpathy/nanochat) (vendored copy without nested `.git`); see [`code/README.md`](code/README.md). Use `uv` as in upstream.
- **`paper`** — preprint [`paper/main.pdf`](paper/main.pdf) and LaTeX sources (`main.tex`, `references.bib`, NeurIPS style).

## Citation

If you find our work helpful, please cite us.

```bibtex
@misc{kiselev2026curvature,
    title={Curvature-Aligned Probing for Local Loss-Landscape Stabilization},
    author={Kiselev, Nikita and Grabovoy, Andrey},
    year={2026},
    note={Manuscript. NeurIPS submission.}
}
```

## Licence

Our project is MIT licensed. See [LICENSE](LICENSE) for details.
