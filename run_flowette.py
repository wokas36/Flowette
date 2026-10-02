from __future__ import annotations

import argparse
import json
import os
from typing import Optional

import numpy as np
import torch
from rdkit import Chem, RDLogger

import fm_graphex.config as cfg


def optional_int(value):
    """Parse an integer or the string 'None'."""
    if value is None:
        return None

    value = str(value).strip()

    if value.lower() in {"none", "null", ""}:
        return None

    return int(value)
    

def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Flowette/Amortised FGW"
    )
    p.add_argument("--task", choices=["qm9", "zinc", "guacamol", "moses"], default="guacamol")
    p.add_argument("--seed", type=int, default=2)
    p.add_argument("--max-num-graphs", type=optional_int, default=None)

    p.add_argument("--ae-node-dim", type=int, default=64)
    p.add_argument("--ae-hidden-dim", type=int, default=128)
    p.add_argument("--ae-batch-size", type=int, default=32)
    p.add_argument("--ae-epochs", type=int, default=None)
    p.add_argument("--ae-lr", type=float, default=1e-3)
    p.add_argument("--ae-ckpt-dir", default="./checkpoints")
    p.add_argument("--no-load-ae", action="store_true")
    p.add_argument("--force-train-fm", action="store_true",
                   help="Ignore a compatible FM checkpoint and retrain it.")
    p.add_argument("--fm-hidden-dim", type=int, default=128)
    p.add_argument("--fm-layers", type=int, default=None)
    p.add_argument("--fm-batch-size", type=int, default=32)
    p.add_argument("--fm-epochs", type=int, default=None)
    p.add_argument("--fm-lr", type=float, default=1e-3)
    p.add_argument("--lambda-x", type=float, default=0.5)
    p.add_argument("--lambda-e", type=float, default=0.5)
    p.add_argument("--alpha-fgw", type=float, default=0.5)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--beta-end", type=float, default=1.0)
    p.add_argument("--beta-valence", type=float, default=0.5)
    p.add_argument("--beta-atom-hist", type=float, default=0.5)
    p.add_argument(
        "--pairing-strategy",
        choices=["auto", "online", "fixed_pool", "periodic"],
        default="auto",
    )
    p.add_argument("--pairing-refresh-every", type=int, default=5)
    p.add_argument("--num-samples", type=int, default=200)
    p.add_argument("--n-steps", type=int, default=None)
    p.add_argument("--bond-threshold", type=float, default=0.1)
    p.add_argument("--gen-batch-size", type=int, default=32)
    p.add_argument("--out", default="./outputs")
    p.add_argument("--skip-eval", action="store_true")
    return p


def apply_dataset_defaults(args: argparse.Namespace) -> argparse.Namespace:
    large = args.task in {"guacamol"}
    if args.ae_epochs is None:
        args.ae_epochs = 20 if large else 10
    if args.fm_epochs is None:
        args.fm_epochs = 30 if large else 10
    if args.fm_layers is None:
        args.fm_layers = 5 if large else 5
    if args.n_steps is None:
        # Signed-logit integration is more stable with a moderate step count.
        args.n_steps = 100 if large else 200
    if args.pairing_strategy == "auto":
        # Refreshing the source pool matters more for diverse large molecules.
        args.pairing_strategy = "periodic" if large else "fixed_pool"
    return args


def run(args: argparse.Namespace):
    args = apply_dataset_defaults(args)
    RDLogger.DisableLog("rdApp.*")
    cfg.set_seed(args.seed)
    cfg.set_dataset(args.task)
    os.makedirs(cfg.get_root(), exist_ok=True)
    os.makedirs(args.out, exist_ok=True)

    # Import dataset-dependent modules only after set_dataset().
    from fm_graphex.data.graphex_noise import load_cycle_counts
    from fm_graphex.flow_matching.sampling import generate_molecules_from_graphex
    from fm_graphex.flow_matching.training import train_flow_matching_graphex
    from fm_graphex.models.encoder import train_autoencoder
    from fm_graphex.models.velocity_field import VelocityFieldGNNTransformer

    print(
        f"[config] task={args.task} root={cfg.get_root()} "
        f"node_feat_dim={cfg.node_feat_dim()} atom_types={cfg.NUM_ATOM_TYPES} "
        f"formal_charges={cfg.FORMAL_CHARGES if cfg.use_formal_charge() else None} "
        "flow_loss=velocity+endpoint+valence+atom_hist"
    )

    encoder, decoder, train_ds, val_ds, test_ds, max_atoms = train_autoencoder(
        task=args.task,
        root=cfg.get_root(),
        node_dim=args.ae_node_dim,
        hidden_dim=args.ae_hidden_dim,
        batch_size=args.ae_batch_size,
        num_epochs=args.ae_epochs,
        lr=args.ae_lr,
        max_num_graphs=args.max_num_graphs,
        ckpt_dir=args.ae_ckpt_dir,
        load_if_exists=not args.no_load_ae,
        save_after_train=True,
    )

    p_hat = cfg.estimate_edge_density(train_ds)
    graphon = np.full((max_atoms, max_atoms), p_hat, dtype=np.float32)
    atom_prob, atom_z, bond_prob, charge_prob = cfg.estimate_molecular_feature_probs(train_ds)
    if charge_prob is not None:
        charge_prior = {
            str(q): float(charge_prob[i]) for i, q in enumerate(cfg.FORMAL_CHARGES)
        }
        # print(f"[priors] formal_charge={charge_prior}")

    train_cycles = load_cycle_counts(cfg.get_cycles_count_train_path())
    if len(train_cycles) < len(train_ds):
        fill = train_cycles[-1] if train_cycles else {}
        train_cycles = list(train_cycles) + [fill] * (len(train_ds) - len(train_cycles))
    train_cycles = train_cycles[:len(train_ds)]

    suffix = (
        f"_{cfg.FEATURE_VERSION}_shared_qm9_zinc_loss_v1"
        if cfg.use_formal_charge()
        else ""
    )
    fm_ckpt = os.path.join(
        args.ae_ckpt_dir,
        f"vnet_{args.task}_hd{args.fm_hidden_dim}_L{args.fm_layers}{suffix}.pt",
    )

    vnet = None
    if os.path.exists(fm_ckpt) and not args.force_train_fm:
        try:
            vnet = VelocityFieldGNNTransformer(
                node_feat_dim=cfg.node_feat_dim(),
                edge_feat_dim=cfg.NUM_BOND_TYPES,
                hidden_dim=args.fm_hidden_dim,
                num_layers=args.fm_layers,
            ).to(cfg.DEVICE)
            state = torch.load(fm_ckpt, map_location=cfg.DEVICE, weights_only=True)
            vnet.load_state_dict(state)
            vnet.eval()
            print(f"[FM] Loaded compatible checkpoint {fm_ckpt}")
        except (RuntimeError, ValueError, TypeError) as exc:
            print(f"[FM] Incompatible checkpoint; retraining. Details: {exc}")
            vnet = None

    if vnet is None:
        vnet = train_flow_matching_graphex(
            encoder=encoder,
            train_dataset=train_ds,
            val_dataset=val_ds,
            cycle_counts=train_cycles,
            graphon=graphon,
            atom_prob=atom_prob,
            atom_z=atom_z,
            bond_prob=bond_prob,
            formal_charge_prob=charge_prob,
            fm_hidden_dim=args.fm_hidden_dim,
            fm_layers=args.fm_layers,
            batch_size=args.fm_batch_size,
            num_epochs=args.fm_epochs,
            lr=args.fm_lr,
            lambda_x=args.lambda_x,
            lambda_e=args.lambda_e,
            alpha_fgw=args.alpha_fgw,
            grad_clip=args.grad_clip,
            ckpt_path=fm_ckpt,
            beta_end=args.beta_end,
            beta_valence=args.beta_valence,
            beta_atom_hist=args.beta_atom_hist,
            pairing_strategy=args.pairing_strategy,
            pairing_refresh_every=args.pairing_refresh_every,
        )

    node_counts = cfg.get_node_count_list(test_ds)
    test_cycles = load_cycle_counts(cfg.get_cycles_count_test_path())
    generated = generate_molecules_from_graphex(
        vnet=vnet,
        graphon=graphon,
        atom_prob=atom_prob,
        atom_z=atom_z,
        bond_prob=bond_prob,
        formal_charge_prob=charge_prob,
        cycle_counts=test_cycles,
        node_counts=node_counts,
        num_samples=args.num_samples,
        n_steps=args.n_steps,
        bond_threshold=args.bond_threshold,
        gen_batch_size=args.gen_batch_size,
        task=args.task,
    )

    smiles = [Chem.MolToSmiles(m, canonical=True, isomericSmiles=True) for m in generated if m is not None]
    smiles_path = os.path.join(args.out, f"{args.task}_{cfg.FEATURE_VERSION}_samples.smi")
    with open(smiles_path, "w", encoding="utf-8") as f:
        f.write("\n".join(smiles))
    print(f"[samples] valid={len(smiles)}/{len(generated)} path={smiles_path}")

    valid_mols = [m for m in generated if m is not None]
    atom_total = sum(m.GetNumAtoms() for m in valid_mols)
    charged_atoms = sum(
        int(atom.GetFormalCharge() != 0)
        for mol in valid_mols for atom in mol.GetAtoms()
    )
    diagnostics = {
        "task": args.task,
        "feature_version": cfg.FEATURE_VERSION,
        "requested_samples": len(generated),
        "valid_decoded": len(valid_mols),
        "decoder_validity": len(valid_mols) / max(len(generated), 1),
        "mean_atoms_valid": atom_total / max(len(valid_mols), 1),
        "charged_atom_fraction": charged_atoms / max(atom_total, 1),
        "multi_fragment_smiles": sum("." in smi for smi in smiles),
        "flow_loss": "velocity + endpoint + valence + atom-hist",
    }
    diagnostics_path = os.path.join(
        args.out, f"{args.task}_{cfg.FEATURE_VERSION}_diagnostics.json"
    )
    with open(diagnostics_path, "w", encoding="utf-8") as f:
        json.dump(diagnostics, f, indent=2)
    print(f"[diagnostics] {diagnostics_path} -> {diagnostics}")

    metrics = None
    if not args.skip_eval:
        from fm_graphex.evaluation.metrics import dataset_to_smiles_list, eval_and_log
        train_smiles = dataset_to_smiles_list(train_ds, args.task)
        test_smiles = dataset_to_smiles_list(test_ds, args.task)
        _, metrics = eval_and_log(generated, (train_smiles, test_smiles), log=False, nspdk_complexity=6)
    return generated, metrics


def main(argv: Optional[list[str]] = None):
    args = build_arg_parser().parse_args(argv)
    return run(args)


if __name__ == "__main__":
    main()