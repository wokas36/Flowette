from __future__ import annotations

import ast
import importlib.util
from pathlib import Path
import sys
import types

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
TRAINING_FILE = ROOT / "fm_graphex/flow_matching/training.py"


def check_source_contract() -> None:
    source = TRAINING_FILE.read_text(encoding="utf-8")
    tree = ast.parse(source)

    functions = {node.name: node for node in tree.body if isinstance(node, ast.FunctionDef)}
    assert "fm_batch_loss" in functions

    args = {arg.arg for arg in functions["fm_batch_loss"].args.args}
    expected = {"beta_end", "beta_valence", "beta_atom_hist"}
    assert expected.issubset(args)


def install_optional_dependency_stubs() -> None:
    class Data:
        def __init__(self, A=None, X=None, E=None):
            self.A = A
            self.X = X
            self.E = E
            self.num_nodes = 0 if X is None else int(X.size(0))

    class DataLoader:
        def __init__(self, *args, **kwargs):
            pass

    tg = types.ModuleType("torch_geometric")
    tg_data = types.ModuleType("torch_geometric.data")
    tg_loader = types.ModuleType("torch_geometric.loader")
    tg_data.Data = Data
    tg_loader.DataLoader = DataLoader
    tg.data = tg_data
    tg.loader = tg_loader
    sys.modules.setdefault("torch_geometric", tg)
    sys.modules.setdefault("torch_geometric.data", tg_data)
    sys.modules.setdefault("torch_geometric.loader", tg_loader)

    encoder_mod = types.ModuleType("fm_graphex.models.encoder")
    encoder_mod.GINEncoder = type("GINEncoder", (), {})
    sys.modules["fm_graphex.models.encoder"] = encoder_mod

    velocity_mod = types.ModuleType("fm_graphex.models.velocity_field")
    velocity_mod.VelocityFieldGNNTransformer = type(
        "VelocityFieldGNNTransformer", (), {}
    )
    sys.modules["fm_graphex.models.velocity_field"] = velocity_mod

    noise_mod = types.ModuleType("fm_graphex.data.graphex_noise")
    noise_mod.sample_graphex_noise_graph_batch = lambda *args, **kwargs: (_ for _ in ()).throw(
        AssertionError("precomputed pair path should not sample noise")
    )
    sys.modules["fm_graphex.data.graphex_noise"] = noise_mod

    fgw_mod = types.ModuleType("fm_graphex.ot.fgw")
    fgw_mod.batch_fgw_distance_matrix = lambda *args, **kwargs: (_ for _ in ()).throw(
        AssertionError("precomputed pair path should not run FGW")
    )
    sys.modules["fm_graphex.ot.fgw"] = fgw_mod

    dense_mod = types.ModuleType("fm_graphex.flow_matching.dense_graph")
    dense_mod.build_dense_type_graph = lambda g, num_nodes, edge_feat_dim: (
        g.A,
        g.X,
        g.E,
    )
    sys.modules["fm_graphex.flow_matching.dense_graph"] = dense_mod


def check_finite_gradient() -> None:
    sys.path.insert(0, str(ROOT))
    install_optional_dependency_stubs()

    import fm_graphex.config as cfg
    from torch_geometric.data import Data

    cfg.set_dataset("guacamol")
    n = 3
    dx = cfg.node_feat_dim()
    de = cfg.NUM_BOND_TYPES
    device = cfg.DEVICE

    def node_features(atom_indices, charge_indices):
        X = torch.zeros((n, dx), device=device)
        for i, (atom_idx, charge_idx) in enumerate(zip(atom_indices, charge_indices)):
            X[i, atom_idx] = 1.0
            X[i, cfg.NUM_ATOM_TYPES + charge_idx] = 1.0
        return X

    carbon = cfg.ATOM_TYPES.index("C")
    nitrogen = cfg.ATOM_TYPES.index("N")
    neutral = cfg.formal_charge_to_index(0)
    positive = cfg.formal_charge_to_index(1)

    A0 = torch.zeros((n, n), device=device)
    A1 = torch.tensor(
        [[0.0, 1.0, 0.0], [1.0, 0.0, 1.0], [0.0, 1.0, 0.0]],
        device=device,
    )
    X0 = node_features([carbon, carbon, carbon], [neutral, neutral, neutral])
    X1 = node_features([carbon, nitrogen, carbon], [neutral, positive, neutral])
    E0 = torch.zeros((n, n, de), device=device)
    E1 = torch.zeros((n, n, de), device=device)
    E1[0, 1, 0] = E1[1, 0, 0] = 1.0
    E1[1, 2, 0] = E1[2, 1, 0] = 1.0

    g0 = Data(A0, X0, E0)
    g1 = Data(A1, X1, E1)

    spec = importlib.util.spec_from_file_location("shared_loss_module", TRAINING_FILE)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)

    class TinyVelocity(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.scale = torch.nn.Parameter(torch.tensor(0.25, device=device))

        def forward(self, A, X, E, t):
            return self.scale * (A + 0.1), self.scale * (X + 0.1), self.scale * (E + 0.1)

    vnet = TinyVelocity()
    loss = module.fm_batch_loss(
        vnet=vnet,
        encoder=object(),
        graphs_tgt=[g1],
        cycle_counts=[{}],
        graphon=np.zeros((n, n), dtype=np.float32),
        atom_prob=np.ones(cfg.NUM_ATOM_TYPES) / cfg.NUM_ATOM_TYPES,
        atom_z=np.arange(cfg.NUM_ATOM_TYPES),
        bond_prob=np.ones(de) / de,
        formal_charge_prob=np.ones(len(cfg.FORMAL_CHARGES)) / len(cfg.FORMAL_CHARGES),
        max_atoms=n,
        lambda_x=0.5,
        lambda_e=0.5,
        beta_end=1.0,
        beta_valence=0.5,
        beta_atom_hist=0.5,
        precomputed_pairs=[(g0, g1, np.eye(n, dtype=np.float32))],
    )
    assert loss is not None and torch.isfinite(loss)
    loss.backward()
    assert vnet.scale.grad is not None and torch.isfinite(vnet.scale.grad)


if __name__ == "__main__":
    check_source_contract()
    check_finite_gradient()
    print("Shared-loss contract and finite-gradient checks passed.")