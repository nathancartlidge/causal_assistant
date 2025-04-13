"""A set of methods to somewhat simplify the process of causal bootstrapping"""
import re
import inspect
import warnings

from typing import Any, Literal, Union

import numpy as np
import pandas as pd

import causalBootstrapping as cb
from distEst_lib import MultivarContiDistributionEstimator as MCDE


def _find_primed_features(function_string):
    """
    Finds features which exist in a 'primed' state (typically just the cause var?)
    :param function_string: The probability function string derived from causal analysis
    :return: A dictionary mapping unprimed to primed features
    """
    primed_features = set(re.compile("([A-z]+'+)").findall(function_string))
    primed_feature_map = {p.replace("'", ""): p for p in primed_features}
    return primed_feature_map


def make_data_map(function_string, kernel_used: bool,
                  **features: Union[np.ndarray, pd.DataFrame, tuple[np.ndarray, int]]):
    """
    Creates the 'data' parameter for general causal bootstrapping
    :param function_string: The probability function string derived from causal analysis
    :param features: All features to be included in the data map. Dataframes will have index inserted for preservation
    :return: The 'data' argument, properly formatted
    """
    for feature in features:
        if isinstance(features[feature], tuple):
            features[feature] = features[feature][0]

        if isinstance(features[feature], pd.DataFrame):
            # insert the index as a value, so it doesn't get lost in the deconfound
            features[feature] = features[feature].reset_index().values

    primed_features = _find_primed_features(function_string)

    for feature, primed_feature in primed_features.items():
        if kernel_used:
            features[primed_feature] = features[feature]
        else:
            features[primed_feature] = features.pop(feature)

    return features


def _find_required_distributions(function_string) -> tuple[set[str], dict[str, list[str]]]:
    """
    Extracts distributions from the function string
    :param function_string: The probability function string derived from causal analysis
    :return: A dictionary mapping distribution names to required variables
    """
    # work out the required distributions. This is quite easy, as the estimation only returns probabilies of one shape
    dist_matcher = re.compile(r"P\(([A-z',]*)\)")
    distributions = dist_matcher.findall(function_string)

    # two simplification stages here:
    #  1. remove 's (as we ignore them for distribution estimation purposes?)
    #  2. split by comma to find individual parameters we need - note that these might be duplicated
    required_dists = {d: d.replace("'", "_prime").split(",") for d in distributions}

    # find all required features
    required_features = set(var for dist in required_dists.values() for var in dist)

    return required_features, required_dists


def _make_dist(required_values: list[str], bins: dict[str, int], features: dict[str, Any],
               fit_method: Literal['kde', 'hist'], estimator_kwargs: Union[dict[str, Any], None] = None):
    """
    Make a distribution
    :param required_values: the features to include in the distribution
    :param bins: dictionary mapping feature names to a bin count
    :param features: dictionary mapping feature names to features
    :param fit_method: The fitting method to use ('kde' or 'hist')
    :param estimator_kwargs: extra kwargs for the estimator
    :return: The distribution method
    """
    data_bins = [bins[r] for r in required_values]
    if len(required_values) == 1:
        data_fit = features[required_values[0].replace("_prime", "")]
    else:
        data_fit_values = [features[r.replace("_prime", "")] for r in required_values]
        data_fit = np.hstack(data_fit_values)
    # create the estimator
    estimator = MCDE(data_fit=data_fit, n_bins=data_bins)
    # fit the estimator
    if fit_method == "kde":
        pdf, probs = estimator.fit_kde(**estimator_kwargs)
    elif fit_method == "hist":
        pdf, probs = estimator.fit_histogram(**estimator_kwargs)
    else:
        raise ValueError("Unrecognised fit method")

    # make the lambda function as per the specification
    pdf_method = lambda **kwargs: pdf(list(kwargs.values())[0]) \
        if len(kwargs) == 1 else pdf(list(kwargs.values()))

    # unfortunately, because the inspection method used is a little annoying, we have to make sure that we fix
    #  the signature as well
    params = [inspect.Parameter(name=r, kind=inspect.Parameter.POSITIONAL_OR_KEYWORD) for r in required_values]
    pdf_method.__signature__ = inspect.Signature(parameters=params)

    return pdf_method


def make_dist_map(function_string, fit_method: Literal['kde', 'hist'] = "kde",
                  estimator_kwargs: Union[dict, None] = None, **features: Union[np.ndarray, tuple[np.ndarray, int]]):
    """
    Creates the 'dist_map' parameter for general causal bootstrapping
    :param function_string: The probability function string derived from causal analysis
    :param fit_method: The method to estimate distributions. Can be either 'hist' or 'kde'
    :param features: All features involved in the de-confounding process. These can either be just the raw feature (for
                     categorical data) or a tuple of the data and the number of bins requested (for continuous data).
    :return: A dictionary mapping probability functions to lambda methods representing their distributions
    """
    required_features, required_dists = _find_required_distributions(function_string)

    # validation: ensure that all required values have been provided
    missing_values = [x for x in required_features if x not in features and not x.endswith("_prime")]
    assert len(missing_values) == 0, f"Not all values provided! {missing_values}"

    # split out values and bin counts from arguments
    bins = {p: 0 for p in required_features}
    for key in features:
        if isinstance(features[key.replace("_prime", "")], tuple):
            # bins included
            bins[key] = features[key][1]
            features[key] = features[key][0]

    # new type of features, now that we have removed all the tuples
    features: dict[str, np.ndarray]

    if estimator_kwargs is None:
        estimator_kwargs = {}

    distributions: dict[str, callable] = {}
    for required_key, required_values in required_dists.items():
        try:
            # note that there may be repetitions here!
            # we are okay with that, it'll just waste a bit of compute
            pdf_method = _make_dist(required_values, bins, features, fit_method, estimator_kwargs)
            distributions[required_key] = pdf_method
        except ValueError:
            print("Required values were", required_values)
            raise

    return distributions


def calc_weights(cause_var, data_map, dist_map, features, weight_func, kernel_used=False):
    """
    Compute causal weights for a given set of distributions
    Adapted from general_causal_bootstrapping_simple in causalBootstrapping
    """
    intv_cause = f"intv_{cause_var}"
    kernel = eval(f"lambda {intv_cause}, {cause_var}: 1 if {intv_cause}=={cause_var} else 0")
    N = features[cause_var].shape[0]
    w_func = weight_func(dist_map=dist_map, N=N, kernel=kernel)
    unique_causes = np.unique(features[cause_var])
    weights = np.zeros((N, len(unique_causes)), dtype=np.float64)

    for i, y in enumerate(unique_causes):
        weights[:, i] = cb.weight_compute(weight_func=w_func,
                                          data=data_map,
                                          intv_var={intv_cause if kernel_used else cause_var: [y for _ in range(N)]})

    all_weights_equal = np.std(weights, axis=0).sum() < 1e-10
    if all_weights_equal:
        warnings.warn("All weights are equal! Check your data")

    return weights, all_weights_equal


def _bootstrap(weight_func: callable, function_string: str, cause_var: str, effect_var: str,
               fit_method: Literal['kde', 'hist'], steps: int = 50,
               **features: Union[np.ndarray, pd.DataFrame, tuple[np.ndarray[Any, Any], int]]):
    # assume effect data will be a dataframe, so will have an index?
    if not isinstance(features[effect_var], pd.DataFrame):
        # auto-cast
        features[effect_var] = pd.DataFrame(features[effect_var])

    # not a massive fan of this :(
    kernel_used = re.match(rf".*(K\({cause_var},{cause_var}'+\)).*", function_string) is not None

    data_map = make_data_map(function_string, kernel_used=kernel_used, **features)
    dist_map = make_dist_map(function_string, fit_method=fit_method, **features)

    weights, _ = calc_weights(cause_var, data_map, dist_map, features, weight_func, kernel_used=kernel_used)

    if kernel_used:
        ivn = cause_var
    else:
        ivn = next(d for d in data_map.keys() if d.startswith(cause_var))

    bootstraps = []
    with warnings.catch_warnings():
        warnings.simplefilter(action="ignore", category=RuntimeWarning)
        for _ in range(steps):
            bootstrap_data, _ = cb.bootstrapper(
                data=data_map, weights=weights, mode="robust",
                intv_var_name_in_data=[ivn]
            )
            bootstraps.append(bootstrap_data)

    cb_data = {}
    for key in bootstraps[0]:
        cb_data[key] = np.vstack([d[key] for d in bootstraps])

    original_df = features[effect_var]
    levels = original_df.index.nlevels
    if levels > 1:
        idx = pd.MultiIndex.from_tuples(cb_data[effect_var][:, 0:levels].tolist(),
                                        names=original_df.index.names)
    else:
        idx = pd.Index(cb_data[effect_var][:, 0], name=original_df.index.name)

    X = pd.DataFrame(cb_data[effect_var][:, levels:], index=idx, columns=original_df.columns)
    y = pd.DataFrame(cb_data[cause_var], index=idx)

    return X, y


def validate_causal_graph(causal_graph: str | None, cause_var: str = "y", effect_var: str = "X") -> str:
    """Validates that a causal graph is correctly configured."""
    if causal_graph is None:
        # non-causal bootstrapping!
        causal_graph = f"{cause_var};{effect_var};{cause_var}->{effect_var};"

    assert cause_var in causal_graph, f"cause var. '{cause_var}' does not appear in the causal graph?"
    assert effect_var in causal_graph, f"effect var. '{cause_var}' does not appear in the causal graph?"

    # todo: remove comments from the causal graph
    return causal_graph


def validate_causal_features(effect_var: str, **features: Union[np.ndarray, pd.DataFrame, tuple[np.ndarray, int]]):
    """Validate that each causal feature is of the correct shape etc"""
    length = features[effect_var].shape[0]

    for var in features:
        if var == effect_var:
            continue

        f = features[var]
        if isinstance(f, tuple):
            f = f[0]

        assert isinstance(f, (np.ndarray, pd.DataFrame)), f"parameter {var} is of type {type(var)}, not expected " \
            "(numpy array / pandas dataframe)!"

        if len(f.shape) == 1 and f.shape[0] == length:
            # todo: automatically fix features, including auto-factorisation
            #       also we should probably raise some kind of warning when we do this?
            # flat array: reshape it for you
            if isinstance(f, np.ndarray):
                features[var] = f.reshape(-1, 1)
        else:
            assert len(f.shape) == 2 and f.shape[0] == length and f.shape[1] == 1, \
                f"feature '{var}' is of wrong shape {f.shape} (should be [{length}, 1])"

        try:
            assert np.isnan(f).sum() == 0, f"feature '{var}' contains NaN values"
        except ValueError:
            raise ValueError(f"feature '{var}' might be of wrong type?")

        assert f.dtype != bool, f"feature '{var}' must not be of type bool"
