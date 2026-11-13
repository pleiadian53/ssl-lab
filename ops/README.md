# ops — remote-training infrastructure

Cloud infra for training ssl-lab models that don't fit in your local environment. It is
**deliberately decoupled** from the SSL library (`ssllab`): nothing here imports
JEPA, and `ssllab` never imports this. Cloud provisioning is necessary
scaffolding for practically useful models, not part of self-supervised learning
itself.

**SkyPilot orchestrates; RunPod is the provider** — `cloud: runpod` is a config
value, not a hard-coded API client, so switching providers is a config change.

## Install

`ops` is an optional helper: nothing in `src/` or `examples/` outside `examples/ops/` depends on it, and you need it only to train on a remote pod or to link shared datasets into `data/`. It is a top-level package of this repository, next to `src/`, and it is not installed: Python finds it whenever the repository root is on the import path. That is automatic for `python -c ...` and `python -m ...` run from the root, the drivers in `examples/ops/` add the root themselves, and the test suite adds it through `pythonpath` in `pyproject.toml`. Its few dependencies come as optional extras of the main project:

```bash
pip install -e '.[ops]'      # PyYAML: compute check and dry runs
pip install -e '.[remote]'   # adds SkyPilot, needed only to launch pods
```

## Workflow

```
compute check ──▶ provision pod ──▶ run job(s) ──▶ fetch results
```

```bash
# 1. What am I on? Do I need a pod?
python examples/ops/ops_compute_check.py

# 2. Dry-run a training job (prints SkyPilot YAML + cost, no cloud calls)
python examples/ops/ops_run_pipeline.py --gpu a40 -- \
    python examples/jepa_basics/01_train_jepa_mnist.py --epochs 50

# 3. Launch for real — the pod is KEPT ALIVE by default (reuse it across sessions).
#    Results rsync back to output/; re-run more jobs on the same pod by passing
#    --cluster ssl-gen-jepa (the job name).
python examples/ops/ops_run_pipeline.py --execute --gpu a40 -- \
    python examples/jepa_basics/01_train_jepa_mnist.py --epochs 50

# 4. When done with all sessions, release the pod (either way):
python examples/ops/ops_run_pipeline.py --execute --teardown --gpu a40 -- <last job>
#   or directly:  sky down ssl-gen-jepa -y
```

> Defaults: `ops_run_pipeline.py` is **dry-run** until `--execute`, and **keeps
> the pod alive** after a run (pass `--teardown` to release it) so you can run
> multiple training sessions on one pod. `ops_provision_cluster.py` is the
> opposite — it **provisions for real** by default (use `--dry-run` to preview).

## Datasets & storage dedup

ssl-lab's own small data (MNIST, SSL-specific) lives in `data/` as real files —
nothing to dedup. For **large datasets shared across projects**, keep a single
physical copy in a project-neutral "data lake" and link only what you use:

```bash
export SSLLAB_DATA_ROOT=~/work/_datalake          # the shared lake (not any one project)
python -c "from ops.datasets import link_dataset; link_dataset('GRCh38')"
# -> data/GRCh38 symlinks into $SSLLAB_DATA_ROOT/GRCh38
```

`SSLLAB_DATA_ROOT` carries the "shared lake" semantics, so ssl-lab never hard-codes
another project's name. Only the datasets you actually reuse get linked, keeping
the dependency explicit. The local input dir is the `data_path` field in
[configs/gpu_config.yaml](configs/gpu_config.yaml) (default `data`).

## Layout

```
ops/
  gpu_runner.py          InfraConfig · build_skypilot_config · print_dry_run · launch · provision/status/down
  hardware.py            detect_hardware · print_report (compute check)
  datasets.py            link_dataset: symlink a dataset from the shared data lake into data/
  configs/
    gpu_config.yaml      infra defaults (GPU, cloud, volume, setup, output paths)
    skypilot/generated/  auto-written SkyPilot task YAMLs
  tests/                 test_config.py · test_datasets.py
examples/ops/            thin drivers: ops_compute_check · ops_run_pipeline · ops_provision_cluster · ops_stage_data
```

## Configure

Edit [configs/gpu_config.yaml](configs/gpu_config.yaml) for defaults, or override
per-run with `--gpu`, `--use-volume`, etc. GPU choices and indicative hourly
rates live in `GPU_SPECS` in [gpu_runner.py](gpu_runner.py) — verify
rates against live RunPod pricing before relying on cost estimates.

## Data

Two pieces handle large inputs. Locally, `datasets.link_dataset` symlinks a dataset from the shared data lake, the directory named by `SSLLAB_DATA_ROOT`, into `data/`, so several projects can share one copy instead of duplicating it. For the pod, `examples/ops/ops_stage_data.py` copies local datasets and checkpoints to the running pod's network volume with rsync, under `/runpod-volume/ssl-lab/`, where they persist across pod lifecycles and the pod's jobs read them.
