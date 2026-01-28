import os
import numpy as np
from rdkit.Chem.Draw import IPythonConsole
from rdkit.Chem import Draw
from rdkit import RDLogger
from IPython.display import display
import pandas as pd
import argparse
from typing import List
from torch_geometric.data import Data

from fm_graphex.config import (
    DEVICE, set_dataset, set_seed, estimate_atom_and_bond_probs, estimate_edge_density,
    get_root, get_node_count_list, get_cycles_count_path, get_nodes_count_path,
    get_cycles_count_train_path, get_cycles_count_test_path
)
from fm_graphex.models.encoder import train_autoencoder
from fm_graphex.data.graphex_noise import load_cycle_counts, load_node_counts
from fm_graphex.flow_matching.training import train_flow_matching_graphex
from fm_graphex.flow_matching.sampling import generate_molecules_from_graphex
from fm_graphex.evaluation.metrics import eval_and_log_ours, dataset_to_smiles_list

RDLogger.DisableLog('rdApp.*')


def build_arg_parser():
    parser = argparse.ArgumentParser(description="Flow Matching + Graphex Molecular Generation")

    # Encoder hyperparameters
    parser.add_argument("--ae_node_dim", type=int, default=64)
    parser.add_argument("--ae_hidden_dim", type=int, default=128)
    parser.add_argument("--ae_batch_size", type=int, default=32)
    parser.add_argument("--num_epochs_ae", type=int, default=100)
    parser.add_argument("--ae_lr", type=float, default=1e-3)
    parser.add_argument("--max_num_graphs", type=int, default=None)
    parser.add_argument("--ae_ckpt_dir", type=str, default="./checkpoints")

    # Flow Matching (FM) hyperparameters
    parser.add_argument("--fm_hidden_dim", type=int, default=128)
    parser.add_argument("--fm_node_dim", type=int, default=64)
    parser.add_argument("--fm_layers", type=int, default=5)
    parser.add_argument("--fm_batch_size", type=int, default=32)
    parser.add_argument("--num_epochs_fm", type=int, default=200)
    parser.add_argument("--fm_lr", type=float, default=1e-3)
    parser.add_argument("--lambda_x", type=float, default=0.5)
    parser.add_argument("--lambda_e", type=float, default=0.5)
    parser.add_argument("--alpha_fgw", type=float, default=0.5)
    parser.add_argument("--beta_end", type=float, default=1.0)
    parser.add_argument("--beta_valence", type=float, default=0.5)
    parser.add_argument("--beta_atom_hist", type=float, default=0.5)
    parser.add_argument("--beta_bond_hist", type=float, default=0.2)

    # Sampling hyperparameters
    parser.add_argument("--num_samples", type=int, default=200)
    
    parser.add_argument("--n_steps", type=int, default=5)
    parser.add_argument("--bond_threshold", type=float, default=0.1)
    parser.add_argument("--gen_batch_size", type=int, default=32)

    parser.add_argument("--seed", type=int, default=2)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--task", type=str, default="qm9", choices=["qm9", "zinc", "guacamol", "moses"])

    return parser


def main():
    parser = build_arg_parser()
    args = parser.parse_args("")
    
    set_seed(args.seed)
    set_dataset(args.task)
    os.makedirs(get_root(), exist_ok=True)

    # Encoder pretraining
    encoder, decoder, train_dataset, val_dataset, test_dataset, max_atoms = train_autoencoder(
        task=args.task,
        root=get_root(),
        node_dim=args.ae_node_dim,
        hidden_dim=args.ae_hidden_dim,
        batch_size=args.ae_batch_size,
        num_epochs=args.num_epochs_ae,
        lr=args.ae_lr,
        max_num_graphs=args.max_num_graphs,
        ckpt_dir=args.ae_ckpt_dir,
        load_if_exists=True,
        save_after_train=True,
    )

    # graphon = np.random.rand(max_atoms, max_atoms) # 1000
    # graphon[:, :] = 0.2
    
    p_hat = estimate_edge_density(train_dataset)
    graphon = np.full((max_atoms, max_atoms), fill_value=p_hat, dtype=np.float32)
    
    train_cycle_counts = load_cycle_counts(get_cycles_count_train_path())
    train_cycle_counts = train_cycle_counts[:len(train_dataset)]

    print(f"[Data] train={len(train_dataset)} val={len(val_dataset)} test={len(test_dataset)}")
    
    # After train_dataset, val_dataset, test_dataset are created:
    atom_prob, atom_z, bond_prob = estimate_atom_and_bond_probs(train_dataset)

    # Train Flow Matching with graphex noise + FGW alignment
    fm_ckpt = f"./checkpoints/vnet_{args.task}_hd{args.fm_hidden_dim}_L{args.fm_layers}.pt"

    vnet = train_flow_matching_graphex(
        encoder=encoder,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        cycle_counts=train_cycle_counts,
        graphon=graphon,
        atom_prob=atom_prob,
        atom_z=atom_z,
        bond_prob=bond_prob,
        fm_hidden_dim=args.fm_hidden_dim,
        fm_layers=args.fm_layers,
        batch_size=args.fm_batch_size,
        num_epochs=args.num_epochs_fm,
        lr=args.fm_lr,
        lambda_x=args.lambda_x,
        lambda_e=args.lambda_e,
        alpha_fgw=args.alpha_fgw,
        grad_clip=args.grad_clip,
        ckpt_path=fm_ckpt,
        beta_end=args.beta_end,
        beta_valence=args.beta_valence,
        beta_atom_hist=args.beta_atom_hist,
        beta_bond_hist=args.beta_bond_hist,
        beta_deg=0.1,
        beta_cycle=0.05,
    )

    test_node_counts = get_node_count_list(test_dataset)
    test_cycle_counts = load_cycle_counts(get_cycles_count_test_path())
    test_cycle_counts = test_cycle_counts[:args.num_samples]
    
    # Generate molecules from graphex noise
    gen_mols = generate_molecules_from_graphex(
        vnet=vnet,
        graphon=graphon,
        atom_prob=atom_prob,
        atom_z=atom_z,
        bond_prob=bond_prob,
        cycle_counts=test_cycle_counts,
        node_counts=test_node_counts,
        num_samples=args.num_samples,
        n_steps=args.n_steps,
        bond_threshold=args.bond_threshold,
        gen_batch_size=args.gen_batch_size,
        task=args.task,
    )

    # Evaluate
    train_smiles = dataset_to_smiles_list(train_dataset, args.task)
    test_smiles  = dataset_to_smiles_list(test_dataset, args.task)

    valid_mols_dict, metrics_dict = eval_and_log_ours(
        mols=gen_mols,
        smiles_pair=(train_smiles, test_smiles),
        log=False,
    )

    # Show generated molecules
    IPythonConsole.drawOptions.maxMols = 1000

    for k, k_mols in valid_mols_dict.items():
        if not k_mols: # skip empty lists
            continue
        try:
            # Draw grid of molecules for this key
            img = Draw.MolsToGridImage(k_mols, molsPerRow=10, subImgSize=(200, 200), maxMols=1000,
                                       returnPNG=False, useSVG=False)
            timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
            out_path = f"images/{args.task}_epoch_all_{timestamp}_{k}.png"
            os.makedirs("images", exist_ok=True)
            img.save(out_path)
            print(f"Saved image at -> {out_path}")
        except Exception as e:
            print(f"Failed to draw/save molecules for key {k}: {e}")
            
            
if __name__ == "__main__":
    main()