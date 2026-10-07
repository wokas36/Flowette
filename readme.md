# Flowette: Flow Matching with Graphette Priors for Graph Generation

This is a Python 3 implementation of Flowette with amortised FGW coupling.

```text
Flowette/
├── run_flowette.py
├── run_synthetic.py
├── fm_graphex/
│   ├── config.py
│   ├── models/
│   │   ├── encoder.py
│   │   └── velocity_field.py
│   ├── ot/
│   │   └── fgw.py
│   ├── data/
│   │   ├── graphex_noise.py
│   │   ├── synthetic.py
│   │   └── synthetic_graphex_noise.py
│   ├── flow_matching/
│   │   ├── dense_graph.py
│   │   ├── training.py
│   │   ├── sampling.py
│   │   ├── synthetic_training.py
│   │   └── synthetic_sampling.py
│   └── evaluation/
│       ├── metrics.py
│       └── synthetic_metrics.py
└── tests/
```

## Installation

Python 3.11 is recommended.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

For CUDA, install the PyTorch and PyG builds appropriate for your CUDA version before installing the remaining requirements.

## Molecular graph experiments

We evaluate Flowette on **QM9, ZINC, MOSES, and GUACAMOL** molecular graph benchmarks.

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

## Files description

- `fm_graphex/config.py`: dataset specifications vocabulary.
- `fm_graphex/data/common_utils.py`: molecule-to-PyG conversion.
- `fm_graphex/features/molecule_like_features.py`: flowette source features.
- `fm_graphex/models/encoder.py`: GIN graph autoencoder.
- `fm_graphex/flow_matching/training.py`: FGW pairing with one shared objective for all datasets.
- `fm_graphex/flow_matching/sampling.py`: sampling and Euler integration.
- `run_flowette.py`: command-line end-to-end runner.

## Synthetic graph experiments

We evaluate Flowette on **Ego-small, Tree, and SBM** synthetic graph benchmarks.

```bash
python run_synthetic.py \
  --task tree \
  --edge-budget-mode sample \
  --require-connected \
  --num-samples 40 \
  --n-steps 50
```

### Synthetic-only files

- `fm_graphex/data/synthetic.py`: SPECTRE-style Ego/Tree/SBM loading and graph conversion.
- `fm_graphex/data/synthetic_graphex_noise.py`: Ego/Tree graphette priors and fitted SBM source prior.
- `fm_graphex/flow_matching/synthetic_training.py`: synthetic velocity + endpoint objective; calls the shared FGW alignment utilities.
- `fm_graphex/flow_matching/synthetic_sampling.py`: size-conditioned edge budgets and synthetic decoding; calls shared `integrate_flow`.
- `fm_graphex/evaluation/synthetic_metrics.py`: Deg/Clus/Orb + VUN evaluation and graph visualisation.
- `run_synthetic.py`: one CLI main for Ego-small, Tree, and SBM.


## Citation

Please cite our paper if you use this code in your research work.

```
@inproceedings{wijesinghe2026neurips,
  title={Flowette: Flow Matching with Graphette Priors for Graph Generation},
  author={Asiri Wijesinghe, Sevvandi Kandanaarachchi, Daniel M Steinberg, Cheng Soon Ong},
  booktitle={40th Conference on Neural Information Processing Systems (NeurIPS)},
  year={2026}
}
```

## License

MIT License