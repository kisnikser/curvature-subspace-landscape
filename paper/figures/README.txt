Figures referenced in main.tex:

  losses_assumption.pdf            — gradient mismatch / Assumption
  criterion_abs.pdf              — one-point criterion (schematic)
  criterion_squared.pdf          — full-space mean-squared criterion (schematic)
  criterion_squared_subspace.pdf — subspace mean-squared criterion (schematic)

  scaling_delta_loglog.pdf                    — primary log-log Δ_1 and Δ_2 vs k
  scaling_delta2_subspace_D10.pdf             — primary Δ_2 vs Δ_2^(D), D=10
  ablation_sigma_sweep.pdf                    — primary Δ_2 under sigma sweep
  ablation_random_vs_hessian_D10.pdf          — top-Hessian vs random subspace
  ablation_refresh_policy_D10.pdf             — recomputed vs frozen subspace
  eigenspace_drift.pdf                        — overlap between top-D subspaces at k and k+1
  quadratic_alignment.pdf                     — true increment vs quadratic proxy
  setting_comparison_delta2_subspace_D10.pdf  — primary vs stronger validation setting
  validation_loss_vs_k.pdf                    — primary held-out quality vs k

Additional settings (for example `wikitext2_medium`) use the same filenames with
an added suffix, e.g. `scaling_delta_loglog_wikitext2_medium.pdf`.

Regenerate scaling PDFs from the repo root:

  python code/run_experiments.py
  python code/plot_scaling_from_json.py

Outputs land in this directory. Until then, placeholder PDFs may be used so the paper compiles.
