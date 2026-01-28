from typing import Tuple
import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from rdkit import Chem
from rdkit.Chem import rdchem


def sample_random_noise_features(
    n_nodes: int,
    edge_index: torch.Tensor,
    atom_prob: np.ndarray,
    atom_z: np.ndarray,
    bond_prob: np.ndarray,
    aromatic_rate: float = 0.15,
    formal_charge_vals: tuple[int, ...] = (-1, 0, 1),
    formal_charge_probs: tuple[float, ...] = (0.08, 0.84, 0.08),
    formal_charge_clip: float = 2.0,
):
    """
    Create random node/edge features for graphex noise graphs.

    Returns
    -------
    x : (n_nodes, NUM_ATOM_TYPES + EXTRA_NODE_FEATS)
        Node features = [atom one-hot | extra features]
    Z : (n_nodes,)
        Atomic numbers for each node (for chemistry post-processing)
    edge_attr : (E, NUM_BOND_TYPES)
        Bond type one-hot per directed edge in edge_index
    pos : (n_nodes, 3)
        Random 3D positions (if your pipeline expects them)
    """
    import fm_graphex.config as cfg

    # if device is None:
    device = edge_index.device

    # node atom-type one-hot
    type_idx = np.random.choice(cfg.NUM_ATOM_TYPES, size=n_nodes, p=atom_prob)
    type_idx_t = torch.tensor(type_idx, dtype=torch.long, device=device)
    x_atom = F.one_hot(type_idx_t, num_classes=cfg.NUM_ATOM_TYPES).float()

    # atomic number array (Z)
    Z = torch.tensor(atom_z[type_idx], dtype=torch.long, device=device)

    # edge attributes (bond type one-hot)
    E = int(edge_index.size(1))
    if E > 0:
        bond_types = np.random.choice(cfg.NUM_BOND_TYPES, size=E, p=bond_prob)
        bond_types_t = torch.tensor(bond_types, dtype=torch.long, device=device)
        edge_attr = F.one_hot(bond_types_t, num_classes=cfg.NUM_BOND_TYPES).float()
    else:
        edge_attr = torch.empty((0, cfg.NUM_BOND_TYPES), dtype=torch.float32, device=device)

    # extra node features (must match cfg.EXTRA_NODE_FEATS)
    extra_dim = int(getattr(cfg, "EXTRA_NODE_FEATS", 0))
    
    if extra_dim <= 0:
        x = x_atom
    else:
        # Default semantics you mentioned: [formal_charge_norm, is_aromatic]
        extras = []

        # formal_charge_norm (float), normalized/clipped
        # sample {-1,0,1} with given probs -> normalize by clip
        if extra_dim >= 1:
            fc = np.random.choice(formal_charge_vals, size=n_nodes, p=formal_charge_probs)
            fc_t = torch.tensor(fc, dtype=torch.float32, device=device)
            fc_norm = torch.clamp(fc_t, -formal_charge_clip, formal_charge_clip) / formal_charge_clip
            extras.append(fc_norm.view(-1, 1))

        # is_aromatic (0/1 float)
        if extra_dim >= 2:
            arom = (torch.rand(n_nodes, device=device) < aromatic_rate).float()
            extras.append(arom.view(-1, 1))

        # If EXTRA_NODE_FEATS > 2, pad remaining dims with zeros (or add more meaningful ones)
        if extra_dim > len(extras):
            used = sum(t.size(1) for t in extras)
            if used < extra_dim:
                extras.append(torch.zeros((n_nodes, extra_dim - used), device=device))

        x_extra = torch.cat(extras, dim=1)
        x = torch.cat([x_atom, x_extra], dim=1)

    # positions
    pos = torch.randn(n_nodes, 3, device=device)

    # sanity check (this prevents silent shape bugs)
    expected = cfg.NUM_ATOM_TYPES + int(getattr(cfg, "EXTRA_NODE_FEATS", 0))
    if x.size(1) != expected:
        raise RuntimeError(f"Noise x has dim {x.size(1)} but expected {expected} (=NUM_ATOM_TYPES+EXTRA_NODE_FEATS)")

    return x, Z, edge_attr, pos



def igraph_to_rdkit_mol(
    gr,
    Z: torch.Tensor,
    edge_index: torch.Tensor,
    edge_attr: torch.Tensor,
) -> Chem.Mol:
    """
    Build an RDKit Mol from:
      - igraph graph (for node count)
      - atomic numbers Z
      - PyG-style edge_index / edge_attr (one-hot bond types).
    """
    mol = Chem.RWMol()

    # Add atoms
    atom_map = {}
    for i in range(gr.vcount()):
        atomic_num = int(Z[i].item())
        atom = Chem.Atom(atomic_num)
        idx = mol.AddAtom(atom)
        atom_map[i] = idx

    # Add bonds
    E = edge_index.shape[1]
    for e in range(E):
        u = int(edge_index[0, e].item())
        v = int(edge_index[1, e].item())
        bond_class = int(edge_attr[e].argmax().item())

        # Map index -> RDKit bond type
        bond_type = BOND_TYPES[min(bond_class, NUM_BOND_TYPES - 1)]

        # Handle aromatic flag
        if bond_type == rdchem.BondType.AROMATIC:
            mol.AddBond(atom_map[u], atom_map[v], rdchem.BondType.AROMATIC)
            mol.GetAtomWithIdx(atom_map[u]).SetIsAromatic(True)
            mol.GetAtomWithIdx(atom_map[v]).SetIsAromatic(True)
        else:
            mol.AddBond(atom_map[u], atom_map[v], bond_type)

    mol = mol.GetMol()
    Chem.SanitizeMol(mol, catchErrors=True)
    return mol


def igraph_to_pyg(
    gr, 
    atom_prob: np.ndarray,
    atom_z: np.ndarray,
    bond_prob: np.ndarray,
 ) -> Data:
    """
    Convert an igraph.Graph into a PyG Data object with QM9-like
    node/edge features and an RDKit SMILES string.
    """
    # Edge list -> edge_index
    edges = np.array(gr.get_edgelist()).T if gr.ecount() > 0 else np.zeros((2, 0))
    edge_index = torch.tensor(edges, dtype=torch.long)

    n_nodes = gr.vcount()
    x, Z, edge_attr, pos = sample_random_noise_features(n_nodes, edge_index, atom_prob, atom_z, bond_prob)

    data = Data(x=x, z=Z, edge_index=edge_index, edge_attr=edge_attr, pos=pos)
    return data