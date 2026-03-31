Figures referenced in main.tex:

  losses_assumption.pdf            — gradient mismatch / Assumption
  criterion_abs.pdf              — one-point criterion (schematic)
  criterion_squared.pdf          — full-space mean-squared criterion (schematic)
  criterion_squared_subspace.pdf — subspace mean-squared criterion (schematic)

  scaling_delta_loglog.pdf        — log-log Δ_1 and Δ_2 vs k (MNIST MLP)
  scaling_delta2_subspace_D10.pdf — Δ_2 vs Δ_2^(D), D=10

Regenerate scaling PDFs from the repo root:

  python code/run_experiments.py
  python code/plot_scaling_from_json.py

Outputs land in this directory. Until then, placeholder PDFs may be used so the paper compiles.
