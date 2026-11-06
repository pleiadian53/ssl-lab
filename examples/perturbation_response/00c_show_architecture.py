"""Show any component of the model: its operations in order and the tensor shape after each.

A front end to ``ssllab.anatomy``. Name a component and get a table of the steps it takes
(operation, resulting shape, parameters of the layer that ran), grouped into sections such
as "time branch" or "condition branch". Options pick a variant, using the same flag names
as the training scripts, so the architecture of a particular run can be shown by pasting
its flags. Run with no component (or ``--list``) to see every component with a
description; names are forgiving (aliases, case, close-match suggestions).

Every walkthrough replays the real module's forward with its own layers and checks the
result against the real forward on the same inputs; the header reports the difference,
which is zero when the table is a faithful account of the code.

Weights are freshly initialized and inputs are random stand-ins of the right shape:
shapes and operations do not depend on either. Sizes default to the production Norman
configuration; override any with ``--set field=value``. The JEPA components run the
transformer on a full training batch of 256 cells and take 10 to 20 seconds on a CPU;
``--batch 8`` shows the same structure in a few seconds.

Usage
-----
    python examples/perturbation_response/00c_show_architecture.py                  # the catalogue
    python examples/perturbation_response/00c_show_architecture.py velocity-field
    python examples/perturbation_response/00c_show_architecture.py condition --cond-type table --flow-base gaussian
    python examples/perturbation_response/00c_show_architecture.py decoder --zinb --format markdown
    python examples/perturbation_response/00c_show_architecture.py encoder --expand --stats
    python examples/perturbation_response/00c_show_architecture.py pipeline --pert KLF1+MAP2K6 --format chain

Output
------
    printed to the terminal (formats: table, chain, markdown, json)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parents[2] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from ssllab.anatomy import FORMATS, Geometry, get, load_vocab, render, render_catalog, walk

PROG = "python examples/perturbation_response/00c_show_architecture.py"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog=PROG, description="Show a model component's operations and tensor shapes, step by step.",
        formatter_class=argparse.RawDescriptionHelpFormatter, epilog=render_catalog(),
    )
    p.add_argument("component", nargs="?", help="component name or alias (omit to list them)")
    p.add_argument("--list", action="store_true", help="list the components and exit")
    p.add_argument("--format", default="table", choices=FORMATS)
    p.add_argument("--stats", action="store_true", help="add the mean ± std of every tensor (table format)")
    p.add_argument("--expand", action="store_true", help="open up the inside of repeated blocks")
    # Variants: the same flag names as the training scripts.
    p.add_argument("--pert", default=None, help="perturbation name (default: a held-out combination)")
    p.add_argument("--flow-base", default="control", choices=["control", "gaussian"])
    p.add_argument("--cond-type", default="geneset", choices=["geneset", "table"])
    p.add_argument("--compose", default="additive", choices=["additive", "deepsets"])
    p.add_argument("--guidance", type=float, default=1.0)
    p.add_argument("--t", type=float, default=0.37, help="flow time for time-embedding and velocity-field")
    p.add_argument("--zinb", action="store_true")
    p.add_argument("--anchored-mean", action="store_true")
    p.add_argument("--state-dispersion", action="store_true")
    p.add_argument("--encoder-cond", action="store_true", help="cVAE encoder reads the condition")
    p.add_argument("--mode", default="generate", choices=["generate", "train"], help="cVAE path")
    p.add_argument("--reg-coef", type=float, default=0.04, help="VICReg weight in jepa-step")
    # Sizes and data.
    p.add_argument("--n", type=int, default=None, help="generated cells per perturbation (default 200)")
    p.add_argument("--batch", type=int, default=None, help="training batch (default 256)")
    p.add_argument("--set", action="append", default=[], metavar="FIELD=VALUE",
                   help="override any size, e.g. --set embed_dim=128 --set n_tokens=100 (repeatable)")
    p.add_argument("--data-dir", default="data")
    p.add_argument("--artifact", default="norman2019")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.list or not args.component:
        print(render_catalog(prog=PROG))
        return
    try:
        spec = get(args.component)
    except KeyError as e:
        raise SystemExit(e.args[0])

    overrides = dict(kv.split("=", 1) for kv in args.set)
    if args.n is not None:
        overrides["n"] = args.n
    if args.batch is not None:
        overrides["batch"] = args.batch
    try:
        geo = Geometry().with_overrides(overrides)
    except KeyError as e:
        raise SystemExit(e.args[0])

    try:
        w = walk(spec.name, geometry=geo, vocab=load_vocab(args.data_dir, args.artifact), seed=args.seed,
                 pert=args.pert, t=args.t, flow_base=args.flow_base, cond_type=args.cond_type,
                 compose=args.compose, guidance=args.guidance, zinb=args.zinb, anchored_mean=args.anchored_mean,
                 state_dispersion=args.state_dispersion, encoder_cond=args.encoder_cond, mode=args.mode,
                 reg_coef=args.reg_coef, expand=args.expand)
    except KeyError as e:                                       # unknown perturbation or geometry field
        raise SystemExit(e.args[0])
    print(render(w, args.format, stats=args.stats))


if __name__ == "__main__":
    main()
