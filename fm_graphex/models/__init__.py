from importlib import import_module

_EXPORTS = {
    "GINEncoder": (".encoder", "GINEncoder"),
    "SimpleGraphDecoder": (".encoder", "SimpleGraphDecoder"),
    "atom_targets_from_data": (".encoder", "atom_targets_from_data"),
    "charge_targets_from_data": (".encoder", "charge_targets_from_data"),
    "build_edge_targets": (".encoder", "build_edge_targets"),
    "train_autoencoder": (".encoder", "train_autoencoder"),
    "VelocityLayer": (".velocity_field", "VelocityLayer"),
    "VelocityFieldGNNTransformer": (".velocity_field", "VelocityFieldGNNTransformer"),
}

__all__ = list(_EXPORTS)


def __getattr__(name):
    if name not in _EXPORTS:
        raise AttributeError(name)
    module_name, attr_name = _EXPORTS[name]
    value = getattr(import_module(module_name, __name__), attr_name)
    globals()[name] = value
    return value