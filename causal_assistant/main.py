import io
import contextlib
from typing import Literal

import numpy as np
import pandas as pd
import warnings

from causal_assistant.utils import _bootstrap
from causal_assistant.helper import validate_causal_graph, validate_causal_features

# cb appears to raise warnings in some cases
with warnings.catch_warnings(category=SyntaxWarning):
    warnings.simplefilter(action="ignore", category=SyntaxWarning)
    import causalBootstrapping as cb

# detect jupyter notebooks, for fancy math rendering
try:
    get_ipython()  # noqa
    from IPython.display import Math
except NameError:
    Math = lambda x: x


def bootstrap(causal_graph: str, cause_var: str = "y", effect_var: str = "X",
              steps: int = 50, fit_method: Literal['hist', 'kde'] = "kde",
              **features: np.ndarray | pd.DataFrame | tuple[np.ndarray, int | list]):
    """
    Perform a repeated causal bootstrapping on the provided data.

    :return: de-confounded X and y (effect and cause) variables, as re-indexed pandas dataframes
    """
    causal_graph = validate_causal_graph(causal_graph, cause_var=cause_var, effect_var=effect_var)
    features, bins = validate_causal_features(effect_var=effect_var, input_features=features)

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

    return _bootstrap(weight_func, function_string, cause_var=cause_var, effect_var=effect_var, steps=steps,
                      fit_method=fit_method, features=features, bins=bins)


def analyse_graph(graph: str, cause_var: str = "y", effect_var: str = "X", print_output: bool = False):
    graph = validate_causal_graph(causal_graph=graph, cause_var=cause_var, effect_var=effect_var)

    # this is a little hacky, but it's the best option we have
    f = io.StringIO()
    with contextlib.redirect_stdout(f):
        cb.general_cb_analysis(causal_graph=graph, cause_var_name=cause_var, effect_var_name=effect_var, info_print=True)
    output = f.getvalue()

    # extract the computed interventional distribution
    intv_dist = output.splitlines()[0].split(":")[1] \
        .replace("|", "\\mid ") \
        .replace("[", "\\left[") \
        .replace("]", "\\right]")

    if print_output:
        print(output)

    return Math(intv_dist)
