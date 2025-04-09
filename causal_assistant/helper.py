import io
import contextlib

import causalBootstrapping as cb


try:
    # detect jupyter notebooks, for fancy math rendering
    get_ipython()  # noqa
    from IPython.display import Math
except NameError:
    Math = lambda x: x


def analyse_graph(graph: str, cause_var: str = "y", effect_var: str = "X", print_output: bool = False):
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
