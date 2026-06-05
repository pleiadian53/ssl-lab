# examples/ops — remote-training drivers

Thin drivers over the [`ssllab-ops`](../../ops/) package. Install it first:

```bash
pip install -e ops/            # for compute-check + dry-run
pip install -e 'ops/[remote]'  # add SkyPilot to launch real pods
```

| Script | Purpose |
|--------|---------|
| `ops_compute_check.py` | Detect hardware (backend, VRAM, RAM, disk); decide local vs pod. |
| `ops_run_pipeline.py` | Dry-run (default) or `--execute` a job on a pod; keeps the pod alive (pass `--teardown` to release); fetch results to `output/`. |
| `ops_provision_cluster.py` | Bring up (real launch by default; A40) / `--dry-run` / `--status` / `--down` a long-running pod. |

Note the asymmetric defaults: `ops_run_pipeline.py` is **dry-run by default**
(`--execute` to launch); `ops_provision_cluster.py` **provisions for real by
default** (`--dry-run` to preview).

Everything after `--` in `ops_run_pipeline.py` is the command run on the pod:

```bash
python examples/ops/ops_run_pipeline.py --gpu a40 -- \
    python examples/jepa_basics/01_train_jepa_mnist.py --epochs 50
```

Without `--execute` it only prints the SkyPilot YAML + cost estimate (no cloud
calls). See [../../ops/README.md](../../ops/README.md) for the full workflow.
