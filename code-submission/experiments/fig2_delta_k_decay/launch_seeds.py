"""
Launch multiple ``run_delta_k_curriculum`` jobs in parallel (one CUDA device per job) with
different ``--seed``, each writing its own JSON and PDFs. Optionally run ``merge_delta_k_curriculum``
when all children exit successfully.

  cd code
  python -m scripts.launch_delta_k_curriculum_seeds \\
      --out-stem .nanochat/reports/my_run \\
      --seeds 0,1,2,3,4,5,6,7 \\
      --gpus 0,1,2,3,4,5,6,7 \\
      --merge-after \\
      -- \\
      --depth 6 --k-max 10000 --n-k-points 20 --train-batch-size 16

Writes ``{out-stem}_seed{S}.json``, ``{out-stem}_seed{S}_criteria.pdf``, ``..._val.pdf``.
With ``--merge-after``, also writes ``{out-stem}_merged.json`` and merged PDFs (unless
``--merge-out-stem`` overrides the merged prefix).
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import threading
from pathlib import Path
from typing import TextIO


def _code_dir() -> Path:
    return Path(__file__).resolve().parent.parent


def _parse_int_list(s: str) -> list[int]:
    return [int(x.strip()) for x in s.split(",") if x.strip()]


def _tee_seed_stdout(proc: subprocess.Popen, log_f: TextIO, seed: int) -> None:
    """Copy child stdout to ``log_f`` and to the parent terminal with a seed prefix."""
    if proc.stdout is None:
        return
    prefix = f"[seed {seed}] "
    for line in iter(proc.stdout.readline, ""):
        log_f.write(line)
        log_f.flush()
        sys.stdout.write(f"{prefix}{line}")
        sys.stdout.flush()


def main() -> None:
    p = argparse.ArgumentParser(
        description="Parallel curriculum runs with different seeds.",
        epilog="Forward run_delta_k_curriculum args after launcher flags, e.g. "
        "--depth 6 --k-max 10000. Optional `-- --depth 6` also works.",
    )
    p.add_argument("--seeds", type=str, default="0,1,2,3,4,5,6,7")
    p.add_argument(
        "--out-stem",
        type=str,
        required=True,
        help="Path prefix without .json; creates {stem}_seed{S}.json and PDFs",
    )
    p.add_argument(
        "--gpus",
        type=str,
        default=None,
        help="Comma-separated device ids (CUDA_VISIBLE_DEVICES per process). Default: 0..n_seeds-1",
    )
    p.add_argument("--merge-after", action="store_true", help="Run merge_delta_k_curriculum if all OK")
    p.add_argument(
        "--merge-out-stem",
        type=str,
        default=None,
        help="Output prefix for merged artifacts (default: {out-stem}_merged)",
    )
    p.add_argument(
        "--merge-figsize",
        type=str,
        default="7,4.5",
        help="Forwarded to merge_delta_k_curriculum",
    )
    args, forwarded = p.parse_known_args()
    if forwarded and forwarded[0] == "--":
        forwarded = forwarded[1:]

    seeds = _parse_int_list(args.seeds)
    if not seeds:
        raise SystemExit("empty --seeds")

    if args.gpus is not None:
        gpus = [x.strip() for x in args.gpus.split(",") if x.strip()]
    else:
        gpus = [str(i) for i in range(len(seeds))]

    if len(gpus) != len(seeds):
        raise SystemExit(f"Need len(--gpus)==len(seeds), got {len(gpus)} vs {len(seeds)}")

    out_stem = Path(args.out_stem)
    out_stem.parent.mkdir(parents=True, exist_ok=True)

    code_dir = _code_dir()
    procs: list[subprocess.Popen] = []
    log_fs: list[TextIO] = []
    threads: list[threading.Thread] = []

    for seed, gpu in zip(seeds, gpus):
        out_json = out_stem.parent / f"{out_stem.name}_seed{seed}.json"
        plot_prefix = out_stem.parent / f"{out_stem.name}_seed{seed}"
        cmd = [
            sys.executable,
            "-m",
            "scripts.run_delta_k_curriculum",
            "--seed",
            str(seed),
            "--out",
            str(out_json),
            "--plot-prefix",
            str(plot_prefix),
            *forwarded,
        ]
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = gpu
        env["PYTHONUNBUFFERED"] = "1"
        log_path = out_stem.parent / f"{out_stem.name}_seed{seed}.log"
        log_f = open(log_path, "w", encoding="utf-8")
        proc = subprocess.Popen(
            cmd,
            cwd=str(code_dir),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        t = threading.Thread(target=_tee_seed_stdout, args=(proc, log_f, seed), daemon=True)
        t.start()
        procs.append(proc)
        log_fs.append(log_f)
        threads.append(t)

    rc = [0] * len(procs)
    for i, proc in enumerate(procs):
        rc[i] = proc.wait()
    for t in threads:
        t.join()
    for log_f in log_fs:
        log_f.close()
    for i in range(len(procs)):
        print(f"seed {seeds[i]} finished with exit code {rc[i]} (log: {out_stem.name}_seed{seeds[i]}.log)")

    if any(r != 0 for r in rc):
        raise SystemExit(f"One or more runs failed: exit codes {rc}")

    if args.merge_after:
        merge_stem = args.merge_out_stem or str(out_stem) + "_merged"
        glob_pat = str(out_stem.parent / f"{out_stem.name}_seed*.json")
        merge_cmd = [
            sys.executable,
            "-m",
            "scripts.merge_delta_k_curriculum",
            "--glob",
            glob_pat,
            "--out-prefix",
            merge_stem,
            "--figsize",
            args.merge_figsize,
        ]
        print("Running:", " ".join(merge_cmd))
        subprocess.run(merge_cmd, cwd=str(code_dir), check=True)


if __name__ == "__main__":
    main()
