"""Lazy dataset and molecular-conversion exports."""
from importlib import import_module

_EXPORTS = {
    "sanitize_mol_safe": (".rdkit_conversions", "sanitize_mol_safe"),
    "sanitize_mol_with_fallback": (".rdkit_conversions", "sanitize_mol_with_fallback"),
    "kekulize_and_remove_hs": (".rdkit_conversions", "kekulize_and_remove_hs"),
    "kekulize_preserve_hs": (".rdkit_conversions", "kekulize_preserve_hs"),
    "data_to_rdkit_mol": (".rdkit_conversions", "data_to_rdkit_mol"),
    "truncate_mol_connected": (".common_utils", "truncate_mol_connected"),
    "truncate_mol_keep_first_atoms": (".common_utils", "truncate_mol_keep_first_atoms"),
    "mol_to_pyg_with_extras": (".common_utils", "mol_to_pyg_with_extras"),
    "process_graph_qm9": (".qm9", "process_graph_qm9"),
    "load_qm9_heavy_dataset": (".qm9", "load_qm9_heavy_dataset"),
    "process_graph_zinc": (".zinc", "process_graph_zinc"),
    "load_zinc_dataset": (".zinc", "load_zinc_dataset"),
    "process_graph_guacamol": (".guacamol", "process_graph_guacamol"),
    "load_guacamol_dataset": (".guacamol", "load_guacamol_dataset"),
    "process_graph_moses": (".moses", "process_graph_moses"),
    "load_moses_dataset": (".moses", "load_moses_dataset"),
    "sample_graph_from_graphon": (".graphex_sampling", "sample_graph_from_graphon"),
    "sample_graphon_from_sparsified_graphon": (".graphex_sampling", "sample_graphon_from_sparsified_graphon"),
    "add_disjoint_ring_components": (".graphex_sampling", "add_disjoint_ring_components"),
    "add_path_based_ring_closures": (".graphex_sampling", "add_path_based_ring_closures"),
    "generate_graph_from_graphex": (".graphex_sampling", "generate_graph_from_graphex"),
    "load_cycle_counts": (".graphex_noise", "load_cycle_counts"),
    "load_node_counts": (".graphex_noise", "load_node_counts"),
    "sample_graphex_noise_graph_single": (".graphex_noise", "sample_graphex_noise_graph_single"),
    "sample_graphex_noise_graph_batch": (".graphex_noise", "sample_graphex_noise_graph_batch"),
}

__all__ = list(_EXPORTS)


def __getattr__(name):
    if name not in _EXPORTS:
        raise AttributeError(name)
    module_name, attr_name = _EXPORTS[name]
    value = getattr(import_module(module_name, __name__), attr_name)
    globals()[name] = value
    return value
