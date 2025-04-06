from typing import Union

import numpy as np
import pandas as pd

import causalBootstrapping as cb


from causal_assistant.utils import _bootstrap, validate_causal_graph, validate_causal_features


def bootstrap(causal_graph: str, cause_var: str, effect_var: str, steps: int = 50,
              **features: Union[np.ndarray, pd.DataFrame, tuple[np.ndarray, int]]):
    """
    Perform a repeated causal bootstrapping on the provided data.

    :return: de-confounded X and y (effect and cause) variables, as re-indexed pandas dataframes
    """
    causal_graph = validate_causal_graph(causal_graph, cause_var=cause_var, effect_var=effect_var)
    validate_causal_features(cause_var=cause_var, **features)

    try:
        weight_func, function_string = cb.general_cb_analysis(
            causal_graph=causal_graph,
            effect_var_name=effect_var,
            cause_var_name=cause_var,
            info_print=False
        )
    except UnboundLocalError as e:
        exc = ValueError("Unable to determine a valid interventional distribution from the provided causal graph")
        raise exc from e

    return _bootstrap(weight_func, function_string, cause_var=cause_var, effect_var=effect_var, steps=steps, **features)
