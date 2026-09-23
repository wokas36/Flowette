from __future__ import annotations

import torch
from rdkit import Chem

import fm_graphex.config as cfg
from fm_graphex.data.common_utils import mol_to_pyg_with_extras
from fm_graphex.data.rdkit_conversions import data_to_rdkit_mol
from fm_graphex.flow_matching.sampling import (
    dense_to_mol,
    expected_num_edges,
    prune_by_valence,
    threshold_adjacency,
)


def representation_roundtrip(task: str) -> None:
    cfg.set_dataset(task)
    mol = Chem.MolFromSmiles("[O-]C(=O)c1cc[nH+]cc1")
    data = mol_to_pyg_with_extras(mol, max_nodes=88 if task == "guacamol" else 27)
    assert data is not None
    assert data.x.size(1) == cfg.node_feat_dim()
    charge_idx = data.x[:, cfg.NUM_ATOM_TYPES:].argmax(dim=-1)
    charges = [cfg.formal_charge_from_index(int(i)) for i in charge_idx]
    assert -1 in charges and 1 in charges
    assert bool((data.edge_attr[:, 3] > 0.5).any())

    restored = data_to_rdkit_mol(data)
    assert restored is not None
    restored_charges = [a.GetFormalCharge() for a in restored.GetAtoms()]
    assert -1 in restored_charges and 1 in restored_charges


def charged_decoder(task: str) -> None:
    cfg.set_dataset(task)
    x = torch.full((1, cfg.node_feat_dim()), -4.0, device=cfg.DEVICE)
    x[0, cfg.ATOM_TYPES.index("N")] = 4.0
    x[0, cfg.NUM_ATOM_TYPES + cfg.formal_charge_to_index(+1)] = 4.0
    a = torch.zeros((1, 1), device=cfg.DEVICE)
    e = torch.zeros((1, 1, cfg.NUM_BOND_TYPES), device=cfg.DEVICE)
    e[..., 0] = 1.0
    mol = dense_to_mol(a, x, e)
    assert mol is not None
    assert mol.GetAtomWithIdx(0).GetFormalCharge() == 1


def pruning_test() -> None:
    cfg.set_dataset("guacamol")
    n = 5
    a = torch.full((n, n), -6.0, device=cfg.DEVICE)
    for i, j, score in [
        (0, 1, 6.0), (1, 2, 5.0), (2, 3, 4.0), (3, 4, 3.0),
        (0, 2, 2.0), (1, 3, 1.0), (0, 4, 0.5),
    ]:
        a[i, j] = a[j, i] = score
    a.fill_diagonal_(-10.0)

    selected = threshold_adjacency(a, n, expected_num_edges(n, "guacamol"))
    expected = min(round(1.40 * n), n * (n - 1) // 2)
    assert int(selected.triu(1).sum()) == expected

    atoms = torch.tensor([cfg.ATOM_TYPES.index("C")] * n, device=cfg.DEVICE)
    charges = torch.tensor([cfg.NEUTRAL_CHARGE_IDX] * n, device=cfg.DEVICE)
    bond_idx = torch.zeros((n, n), dtype=torch.long, device=cfg.DEVICE)
    bond_idx[0, 1] = bond_idx[1, 0] = 2
    pruned = prune_by_valence(selected, atoms, bond_idx, charges, "guacamol")
    assert torch.allclose(pruned, pruned.T)


def main() -> None:
    for task in ("guacamol", "moses"):
        representation_roundtrip(task)
        charged_decoder(task)
    pruning_test()
    print("All OptionA GuacaMol/MOSES smoke tests passed.")


if __name__ == "__main__":
    main()