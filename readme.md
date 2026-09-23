# Flowette: Flow Matching with Graphette Priors for Graph Generation

This is a Python 3 implementation of Flowette with amortised FGW coupling.

## Installation

Python 3.11 is recommended.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

For CUDA, install the PyTorch and PyG builds appropriate for your CUDA version before installing the remaining requirements.

## End-to-end run

```bash
python run_flowette.py \
  --task guacamol \
  --max-num-graphs None \
  --ae-epochs 50 \
  --fm-epochs 500 \
  --fm-layers 5 \
  --pairing-strategy periodic \
  --pairing-refresh-every 2 \
  --n-steps 100 \
  --num-samples 1000 \
  --force-train-fm
```

## Key implementation files

- `fm_graphex/config.py`: dataset specifications vocabulary.
- `fm_graphex/data/common_utils.py`: molecule-to-PyG conversion.
- `fm_graphex/features/molecule_like_features.py`: flowette source features.
- `fm_graphex/models/encoder.py`: GIN graph autoencoder.
- `fm_graphex/flow_matching/training.py`: FGW pairing with one shared objective for all datasets.
- `fm_graphex/flow_matching/sampling.py`: sampling and Euler integration.
- `run_flowette.py`: command-line end-to-end runner.