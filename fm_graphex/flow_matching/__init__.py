from importlib import import_module

_EXPORTS = {
    "fm_batch_loss": (".training", "fm_batch_loss"),
    "train_flow_matching_graphex": (".training", "train_flow_matching_graphex"),
    "integrate_flow": (".sampling", "integrate_flow"),
    "dense_to_mol": (".sampling", "dense_to_mol"),
    "expected_num_edges": (".sampling", "expected_num_edges"),
    "formal_charge_indices_from_X": (".sampling", "formal_charge_indices_from_X"),
    "generate_molecules_from_graphex": (".sampling", "generate_molecules_from_graphex"),
    "prune_by_valence": (".sampling", "prune_by_valence"),
    "threshold_adjacency": (".sampling", "threshold_adjacency"),
    "normalize_adjacency": (".dense_graph", "normalize_adjacency"),
    "relu": (".dense_graph", "relu"),
    "leaky_relu": (".dense_graph", "leaky_relu"),
    "normalised_softplus": (".dense_graph", "normalised_softplus"),
    "build_dense_type_graph": (".dense_graph", "build_dense_type_graph"),
}

__all__ = list(_EXPORTS)


def __getattr__(name):
    if name not in _EXPORTS:
        raise AttributeError(name)
    module_name, attr_name = _EXPORTS[name]
    value = getattr(import_module(module_name, __name__), attr_name)
    globals()[name] = value
    return value