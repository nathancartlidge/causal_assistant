import pandas as pd
import numpy as np


def validate_causal_graph(causal_graph: str | None, cause_var: str = "y", effect_var: str = "X") -> str:
    """Validates that a causal graph is correctly configured."""
    if causal_graph is None:
        # non-causal bootstrapping!
        causal_graph = f"{cause_var};{effect_var};{cause_var}->{effect_var};"

    assert cause_var in causal_graph, f"cause var. '{cause_var}' does not appear in the causal graph?"
    assert effect_var in causal_graph, f"effect var. '{cause_var}' does not appear in the causal graph?"

    # todo: remove comments from the causal graph
    return causal_graph


def validate_causal_features(effect_var: str, input_features: dict[str, np.ndarray | pd.DataFrame | tuple[np.ndarray, int | list]]):
    """Validate that each causal feature is of the correct shape etc"""
    length = input_features[effect_var].shape[0]

    features = {}
    bins = {}

    for var in input_features:
        if var == effect_var:
            features[var] = input_features[var]
            continue

        f = input_features[var]
        b = None
        if isinstance(f, tuple):
            f = input_features[var][0]
            b = input_features[var][1]

        if f.dtype == bool:
            if len(f.shape) == 2:
                assert f.shape[1] == 1, f"parameter {var} is of bool type, but multi-dimensional - unable to factorise!"
                f = np.reshape(f, -1)

            f = pd.factorize(f)[0]

        assert isinstance(f, (np.ndarray, pd.DataFrame)), f"parameter {var} is of type {type(var)}, not expected " \
            "(numpy array / pandas dataframe)!"

        if len(f.shape) == 1 and f.shape[0] == length:
            # todo: automatically fix features, including auto-factorisation
            #       also we should probably raise some kind of warning when we do this?
            # flat array: reshape it for you
            if isinstance(f, np.ndarray):
                f = f.reshape(-1, 1)

        assert len(f.shape) == 2 and f.shape[0] == length, \
            f"feature '{var}' is of wrong shape {f.shape} (should be [{length}, X])"

        if b is not None:
            assert len(b) == f.shape[1], f"bin size ({len(b)}) must match feature size {f.shape[1]} for '{var}'!"
        else:
            b = [0 for _ in range(f.shape[1])]

        try:
            if isinstance(f, (pd.DataFrame, pd.Series)):
                assert np.isnan(f.values).sum() == 0, f"feature '{var}' contains NaN values"
            else:
                assert np.isnan(f).sum() == 0, f"feature '{var}' contains NaN values"
        except ValueError:
            raise ValueError(f"feature '{var}' might be of wrong type?")

        features[var] = f
        bins[var] = b

    return features, bins