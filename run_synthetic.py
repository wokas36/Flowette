from __future__ import annotations

import argparse
import json
import os
from typing import Optional

import numpy as np
import torch

import fm_graphex.config as cfg
from fm_graphex.models.encoder import DenseGINEncoder
from fm_graphex.models.velocity_field import VelocityFieldGNNTransformer
from fm_graphex.data.synthetic import estimate_edge_density, load_spectre_style_dataset, node_count_list
from fm_graphex.evaluation.synthetic_metrics import (
    EvalConfig,
    USING_EXTERNAL_EVAL_HELPER,
    ValidityConfig,
    evaluate_deg_clus_orb_vun,
    visualize_graph_samples,
)
from fm_graphex.data.synthetic_graphex_noise import (
    estimate_sbm_mixture_from_train,
    load_ego_ring_pool,
    sample_graphette_noise_batch,
    sample_sbm_noise_batch_match_targets,
)
from fm_graphex.flow_matching.synthetic_sampling import (
    build_size_conditioned_edge_counts,
    generate_graphs_from_graphex_fm,
)
from fm_graphex.flow_matching.synthetic_training import (
    pretrain_encoder_reconstruct_adj,
    train_flow_matching_synthetic,
)


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Flowette synthetic graph experiments")
    p.add_argument("--task", choices=["ego", "tree", "sbm"], default="ego")
    p.add_argument("--seed", type=int, default=2)
    p.add_argument("--data-root", default="./spectre_data")
    p.add_argument("--out", default="./outputs/synthetic")
    p.add_argument("--checkpoint-dir", default="./checkpoints/synthetic")
    p.add_argument("--force-train-fm", action="store_true")
    p.add_argument("--force-train-encoder", action="store_true")

    p.add_argument("--enc-hidden", type=int, default=64)
    p.add_argument("--enc-out", type=int, default=32)
    p.add_argument("--enc-layers", type=int, default=3)
    p.add_argument("--enc-pretrain-epochs", type=int, default=None)
    p.add_argument("--enc-lr", type=float, default=1e-3)

    p.add_argument("--fm-hidden-dim", type=int, default=128)
    p.add_argument("--fm-layers", type=int, default=None)
    p.add_argument("--fm-batch-size", type=int, default=32)
    p.add_argument("--fm-epochs", type=int, default=None)
    p.add_argument("--fm-lr", type=float, default=1e-3)
    p.add_argument("--lambda-x", type=float, default=0.5)
    p.add_argument("--lambda-e", type=float, default=0.5)
    p.add_argument("--alpha-fgw", type=float, default=None)
    p.add_argument("--beta-end", type=float, default=1.0)
    p.add_argument("--grad-clip", type=float, default=1.0)

    p.add_argument("--num-samples", type=int, default=None)
    p.add_argument("--n-steps", type=int, default=None)
    p.add_argument("--gen-batch-size", type=int, default=32)
    p.add_argument("--edge-budget-mode", choices=["sample", "mean", "median"], default="sample")
    p.add_argument("--require-connected", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--clustering-bins", type=int, default=10)

    p.add_argument("--ring-count-path", default="cycle_count_ego.txt")
    p.add_argument("--use-rings", action=argparse.BooleanOptionalAction, default=None)
    p.add_argument("--remove-ring", action=argparse.BooleanOptionalAction, default=None)
    p.add_argument("--tree-graphon-resolution", type=int, default=1000)
    p.add_argument("--tree-graphon-value", type=float, default=0.2)
    p.add_argument("--sbm-k-min", type=int, default=2)
    p.add_argument("--sbm-k-max", type=int, default=5)
    p.add_argument("--skip-eval", action="store_true")
    p.add_argument("--skip-viz", action="store_true")
    return p


def apply_task_defaults(args: argparse.Namespace) -> argparse.Namespace:
    if args.task == "ego":
        defaults = dict(enc_pretrain_epochs=10, fm_layers=3, fm_epochs=10, alpha_fgw=0.5,
                        num_samples=400, n_steps=200, use_rings=True, remove_ring=False)
    elif args.task == "tree":
        defaults = dict(enc_pretrain_epochs=10, fm_layers=3, fm_epochs=10, alpha_fgw=0.5,
                        num_samples=40, n_steps=50, use_rings=False, remove_ring=True)
    else:
        defaults = dict(enc_pretrain_epochs=3, fm_layers=4, fm_epochs=10, alpha_fgw=1.0,
                        num_samples=400, n_steps=200, use_rings=False, remove_ring=False)
    for key, value in defaults.items():
        if getattr(args, key) is None:
            setattr(args, key, value)
    return args


def _build_prior(args, train_dataset, max_nodes, p_hat):
    sbm_mix = None
    ring_pool = None
    if args.task == "tree":
        graphon = np.full(
            (int(args.tree_graphon_resolution), int(args.tree_graphon_resolution)),
            float(args.tree_graphon_value),
            dtype=np.float32,
        )
    else:
        graphon = np.full((max(int(max_nodes), 2), max(int(max_nodes), 2)), p_hat, dtype=np.float32)

    if args.task == "ego" and args.use_rings:
        ring_pool = load_ego_ring_pool(args.ring_count_path, train_dataset=train_dataset)
    if args.task == "sbm":
        sbm_mix = estimate_sbm_mixture_from_train(
            train_dataset,
            k_min=args.sbm_k_min,
            k_max=args.sbm_k_max,
            seed=args.seed,
        )
    return graphon, ring_pool, sbm_mix


def _preflight(args, train_dataset, graphon, ring_pool, sbm_mix):
    subset = list(train_dataset[: min(8, len(train_dataset))])
    if not subset:
        raise RuntimeError("Training split is empty.")
    if args.task == "sbm":
        sources = sample_sbm_noise_batch_match_targets(sbm_mix, subset)
    else:
        sources = sample_graphette_noise_batch(
            graphon=graphon,
            n_nodes_list=[int(g.num_nodes) for g in subset],
            ring_sizes_pool=ring_pool,
            remove_ring=args.remove_ring,
            task=args.task,
        )
    src = [int(g.num_nodes) for g in sources]
    tgt = [int(g.num_nodes) for g in subset]
    if src != tgt:
        raise RuntimeError(f"Synthetic source-size preflight failed: source={src}, target={tgt}.")
    print(f"[Preflight] exact-size {args.task} source invariant: OK")


def _load_or_train_encoder(args, train_dataset):
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    path = os.path.join(args.checkpoint_dir, f"encoder_{args.task}_h{args.enc_hidden}_o{args.enc_out}_L{args.enc_layers}.pt")
    encoder = DenseGINEncoder(
        in_dim=1,
        hidden_dim=args.enc_hidden,
        out_dim=args.enc_out,
        num_layers=args.enc_layers,
    ).to(cfg.DEVICE)
    if os.path.exists(path) and not args.force_train_encoder:
        try:
            encoder.load_state_dict(torch.load(path, map_location=cfg.DEVICE, weights_only=True))
            encoder.eval()
            print(f"[Encoder] loaded {path}")
            return encoder
        except (RuntimeError, ValueError, TypeError) as exc:
            print(f"[Encoder] incompatible checkpoint, retraining: {exc}")
    encoder = pretrain_encoder_reconstruct_adj(
        encoder,
        train_dataset=train_dataset,
        num_epochs=args.enc_pretrain_epochs,
        lr=args.enc_lr,
        max_graphs=min(len(train_dataset), 2000),
    )
    torch.save(encoder.state_dict(), path)
    print(f"[Encoder] saved {path}")
    return encoder


def _load_or_train_vnet(args, encoder, train_dataset, val_dataset, graphon, ring_pool, sbm_mix):
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    path = os.path.join(args.checkpoint_dir, f"vnet_{args.task}_hd{args.fm_hidden_dim}_L{args.fm_layers}.pt")
    vnet = VelocityFieldGNNTransformer(
        node_feat_dim=1,
        edge_feat_dim=1,
        hidden_dim=args.fm_hidden_dim,
        num_layers=args.fm_layers,
    ).to(cfg.DEVICE)
    if os.path.exists(path) and not args.force_train_fm:
        try:
            vnet.load_state_dict(torch.load(path, map_location=cfg.DEVICE, weights_only=True))
            vnet.eval()
            print(f"[FM] loaded {path}")
            return vnet
        except (RuntimeError, ValueError, TypeError) as exc:
            print(f"[FM] incompatible checkpoint, retraining: {exc}")

    return train_flow_matching_synthetic(
        encoder=encoder,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        graphon=graphon,
        ring_sizes_pool=ring_pool,
        fm_hidden_dim=args.fm_hidden_dim,
        fm_layers=args.fm_layers,
        batch_size=args.fm_batch_size,
        num_epochs=args.fm_epochs,
        lr=args.fm_lr,
        lambda_x=args.lambda_x,
        lambda_e=args.lambda_e,
        alpha_fgw=args.alpha_fgw,
        grad_clip=args.grad_clip,
        ckpt_path=path,
        beta_end=args.beta_end,
        remove_ring=args.remove_ring,
        task=args.task,
        sbm_mix=sbm_mix,
    )


def run(args: argparse.Namespace):
    args = apply_task_defaults(args)
    cfg.set_seed(args.seed)
    os.makedirs(args.out, exist_ok=True)

    train_ds, val_ds, test_ds, max_nodes = load_spectre_style_dataset(
        args.task, root=args.data_root, seed=1234
    )
    p_hat = estimate_edge_density(train_ds)
    edge_counts_by_n = build_size_conditioned_edge_counts(train_ds)
    graphon, ring_pool, sbm_mix = _build_prior(args, train_ds, max_nodes, p_hat)

    print(f"[Data] task={args.task} train={len(train_ds)} val={len(val_ds)} test={len(test_ds)} max_nodes={max_nodes}")
    print(f"[Prior] global_train_density={p_hat:.6f} edge_budget_mode={args.edge_budget_mode}")
    if args.task == "sbm":
        print(f"[SBM] p(k)={sbm_mix.pk}")
    if args.task == "ego":
        print(f"[Ego] ring_pool={0 if ring_pool is None else len(ring_pool)}")

    _preflight(args, train_ds, graphon, ring_pool, sbm_mix)
    encoder = _load_or_train_encoder(args, train_ds)
    vnet = _load_or_train_vnet(args, encoder, train_ds, val_ds, graphon, ring_pool, sbm_mix)

    counts = node_count_list(test_ds)
    if not counts:
        raise RuntimeError("Test split has no graphs.")
    if args.task in {"tree", "sbm"}:
        counts = counts[: min(args.num_samples, len(counts))]

    generated = generate_graphs_from_graphex_fm(
        vnet=vnet,
        graphon=graphon,
        node_counts=counts,
        ring_sizes_pool=ring_pool,
        edge_counts_by_n=edge_counts_by_n,
        fallback_density=p_hat,
        num_samples=args.num_samples,
        n_steps=args.n_steps,
        edge_budget_mode=args.edge_budget_mode,
        gen_batch_size=args.gen_batch_size,
        require_connected=args.require_connected,
        remove_ring=args.remove_ring,
        task=args.task,
        sbm_mix=sbm_mix,
    )

    samples_path = os.path.join(args.out, f"{args.task}_generated_graphs.pt")
    torch.save(generated, samples_path)
    print(f"[Samples] saved {len(generated)} graphs to {samples_path}")

    if not args.skip_viz:
        visualize_graph_samples(
            generated,
            out_dir=os.path.join(args.out, "images"),
            run_name=f"{args.task}_generated",
            max_show=min(36, len(generated)),
            cols=6,
            seed=args.seed,
            node_size=70,
        )

    metrics = None
    if not args.skip_eval:
        n_eval = min(len(test_ds), len(generated))
        metrics = evaluate_deg_clus_orb_vun(
            train_pyg=train_ds,
            test_pyg=test_ds[:n_eval],
            gen_pyg=generated,
            cfg=EvalConfig(
                clustering_bins=args.clustering_bins,
                validity=ValidityConfig(require_connected=args.require_connected, min_nodes=2),
            ),
            dataset=args.task,
        )
        metrics_path = os.path.join(args.out, f"{args.task}_metrics.json")
        
        print("\nFlowette synthetic", args.task, "evaluation")
        print(f"Clus MMD: {metrics['Clus']:.6f}")
        print(f"Orb MMD: {metrics['Orb']:.6f}")
        print(f"VUN: {metrics['VUN']:.6f}")
        print(f"Validity: {metrics['Validity']:.4f}")
        print(f"Uniqueness: {metrics['Uniqueness']:.4f}")
        print(f"Novelty: {metrics['Novelty']:.4f}")
        print(
            f"(n_gen={metrics['n_gen']} | "
            f"n_gen_valid={metrics['n_gen_valid']} | "
            f"n_test_valid={metrics['n_test_valid']})"
        )

        with open(metrics_path, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)
        # print(json.dumps(metrics, indent=2))
        if not USING_EXTERNAL_EVAL_HELPER:
            print("[Eval] eval_helper.py was not supplied; used internal fallback MMDs. "
                  "Place your original eval_helper.py at project root to reproduce its exact metrics.")
        print(f"[Metrics] {metrics_path}")

    return generated, metrics


def main(argv: Optional[list[str]] = None):
    args = build_arg_parser().parse_args(argv)
    return run(args)


if __name__ == "__main__":
    main()