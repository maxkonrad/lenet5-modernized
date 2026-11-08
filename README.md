# lenet5-modernized

LeNet-5 implemented from scratch in NumPy — forward and backprop written by
hand, no autograd — modernized into a small, reproducible Python project.

## Setup

Requires Python 3.11+ and [uv](https://docs.astral.sh/uv/).

```sh
uv sync --extra dev
```

## Data

MNIST is **not** committed. Download it into the git-ignored `data/` dir:

```sh
uv run lenet-download
```

(Existing `*_idx*-ubyte` files placed in `data/` are picked up automatically.)

## Train

```sh
uv run lenet-train --epochs 10
uv run python -m lenet.train --optimizer sgd --lr 0.01 --train-limit 5000
```

Options: `--epochs`, `--batch-size`, `--lr`, `--optimizer {adam,sgd}`,
`--seed`, `--val-size`, `--train-limit`, `--data-dir`, `--checkpoint-dir`,
`--augment`, `--weight-decay`, `--dropout`.
Best/last weights go to `checkpoints/best.npz` and `checkpoints/last.npz`.

Regularization examples:

```sh
# shifts + elastic distortions (default: off, runs stay reproducible)
uv run lenet-train --augment
# decoupled weight decay (AdamW style) + FC dropout
uv run lenet-train --weight-decay 1e-4 --dropout 0.2
```

## Ensemble

Train one model per seed, then average their test-set probabilities:

```sh
uv run lenet-train --seed 0 --checkpoint-dir checkpoints/seed0
uv run lenet-train --seed 1 --checkpoint-dir checkpoints/seed1
uv run lenet-ensemble --checkpoints checkpoints/seed0/best.npz \
    checkpoints/seed1/best.npz
```

## Test / lint

```sh
uv run pytest
uv run ruff check src tests
uv run ruff format --check src tests
```

## Benchmarks (measured, seed 0, full 10k test set)

```text
5k subset,  10 epochs, plain                        -> test 96.06%
60k full,   10 epochs, plain                        -> test 99.24%  (~70s)
60k full,   10 epochs, --augment                    -> test 99.30%  (~2min)
60k full,   10 epochs, --dropout 0.2 --wd 1e-4      -> test 99.23%  (~70s)
60k full,   15 epochs, --augment                    -> test 99.40%  (~3min)
60k full,   15 epochs, --augment --dropout --wd     -> test 99.31%  (~3min)
60k full,   15 epochs, --augment, 3-seed ensemble   -> 99.35 + 99.39 + 99.35 -> 99.47%
2k subset,   3 epochs, 2-model ensemble             -> 86.41% + 83.33% -> 86.60%
```

## Layout

```text
src/lenet/
  layers.py   pure forward/backward helpers (conv, avgpool, relu, softmax-xent)
  model.py    LeNet5 assembly, He init, FC dropout, save/load, evaluate
  optim.py    SGD + Adam (AdamW-style decoupled weight decay)
  data.py     IDX loading with magic validation + downloader
  augment.py  shifts + elastic distortions (dependency-free NumPy)
  train.py    CLI, train/val split, cosine LR decay, checkpointing
  ensemble.py average checkpoint probabilities on the test set
tests/        smoke, numeric-gradient, data, augment, regularization tests
data/         MNIST binaries (ignored)      checkpoints/  weights (ignored)
```

## What changed vs the original scripts

- Data files untracked; loader validates magic numbers and downloads on demand.
- `float64` → `float32`, `N(0, 0.1)` init → He init, FC biases → zeros.
- Monolithic `forward`/`backward` split into reusable layer helpers; the
  duplicated conv-gradient math is now one shared implementation.
- Training: seeded RNG, train/val split, Adam (SGD optional), cosine LR decay,
  `tqdm` progress, best-checkpoint saving, fixed the `%`-format accuracy bug.
- Regularization: `--augment` (shifts + elastic), `--weight-decay` (decoupled,
  weights only), `--dropout` (inverted, FC layers, eval mode disables it).
- `lenet-ensemble` averages member probabilities; beats both members in tests.
- Added `pyproject.toml` (uv), ruff, pytest suite with a numeric gradient
  check, and CI.
