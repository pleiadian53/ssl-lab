# data/

Local input datasets for ssl-lab. **The datasets themselves are not tracked in
git** (too large and/or sensitive) — only this documentation and the folder
structure are shared. See [`.gitignore`](../.gitignore).

## What lives here

- `MNIST/` — auto-downloaded by torchvision on first run
  (`ssllab.data.mnist.get_mnist_dataloaders`); nothing to fetch manually.

## Large / shared datasets (storage dedup)

ssl-lab's own small data lives here as real files. For **large datasets shared
across projects**, do not duplicate them — keep a single physical copy in a
project-neutral "data lake" and symlink only what you use:

```bash
export SSLLAB_DATA_ROOT=/path/to/shared/data-lake
python -c "from ops.datasets import link_dataset; link_dataset('GRCh38')"
# -> data/GRCh38 symlinks into $SSLLAB_DATA_ROOT/GRCh38
```

This keeps the shared lake project-neutral and the dependency explicit. See
[`ops/README.md`](../ops/README.md) for details.
