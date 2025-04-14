import re
import warnings

import pandas as pd
import numpy as np

from causal_assistant.utils import BinFeatureType, FeatureType


def validate_causal_graph(causal_graph: str | None, cause_var: str = "y", effect_var: str = "X") -> str:
    """Detect (and attempt to resolve) common errors in causal graphs"""
    if causal_graph is None:
        # non-causal bootstrapping!
        causal_graph = f"{cause_var};{effect_var};{cause_var}->{effect_var};"

    assert cause_var in causal_graph, f"cause var. '{cause_var}' does not appear in the causal graph?"
    assert effect_var in causal_graph, f"effect var. '{cause_var}' does not appear in the causal graph?"

    # de-indent the graph
    graph_lines = [l.strip() for l in causal_graph.splitlines()]

    # remove comments and blank lines, support newline as a delimiter
    graph_lines = [l for l in graph_lines if l and not l.startswith("#")]
    causal_graph = ";".join(graph_lines).replace(";;", ";")

    # add a final ; if missing
    if not causal_graph.endswith(";"):
        causal_graph += ";"

    # validate that all variables used in the graph are defined
    clauses = causal_graph.split(";")
    variable_finder = re.compile(r"(.*?)<?->(.*)")
    variables = set([
        v.strip()
        for clause in clauses
        for r in variable_finder.findall(clause)
        for v in r
    ])

    missing_vars = [v for v in variables if v not in clauses]
    if missing_vars:
        raise ValueError(f"graph variable(s) '{','.join(missing_vars)}' not defined at top of graph!")

    return causal_graph


def validate_causal_features(effect_var: str, warn_on_cast: bool, input_features: dict[str, BinFeatureType]) \
        -> tuple[dict[str, FeatureType], dict[str, list[int]]]:
    """
    Detect (and attempt to resolve) common errors in causal graphs, split
    :param effect_var: Typically 'X', a variable not used as part of the causal estimation.
    :param warn_on_cast: If true, warnings will be emitted when the features are modified by this method.
    :param input_features: All features required for the graph.
    :return: tuple: feature dict and bin count dict
    """
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

        if isinstance(f, (pd.DataFrame, pd.Series)):
            f = f.values

        if f.dtype == bool or f.dtype == "O":
            if len(f.shape) == 2:
                assert f.shape[1] == 1, f"parameter '{var}' is of bool type, but multi-dimensional - unable to factorise!"
                f = np.reshape(f, -1)

            if warn_on_cast:
                warnings.warn(message=f"automatically factorising '{var}'", category=RuntimeWarning)

            f = pd.factorize(f)[0]

        if not isinstance(f, (np.ndarray, pd.DataFrame, pd.Series)):
            warnings.warn(message=f"parameter '{var}' is of type {type(f)}, this may cause issues!",
                          category=EncodingWarning)

        if len(f.shape) == 1 and f.shape[0] == length:
            # flat array provided: reshape it
            if warn_on_cast:
                warnings.warn(message=f"automatically re-shaping '{var}'", category=RuntimeWarning)

            if isinstance(f, np.ndarray):
                f = f.reshape(-1, 1)

        assert len(f.shape) == 2 and f.shape[0] == length, \
            f"feature '{var}' is of wrong shape {f.shape} (should be [{length}, X])"

        if b is not None:
            if isinstance(b, int):
                b = [b for _ in range(f.shape[1])]
                if f.shape[1] > 1 and warn_on_cast:
                    warnings.warn(message=f"automatically vectorising bins for '{var}'", category=RuntimeWarning)

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