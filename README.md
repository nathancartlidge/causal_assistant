# Causal Assistant
This library is a semi-wrapper around Jianqiao Mao's `causalBootstrapping` [package](https://github.com/JianqiaoMao/CausalBootstrapping)
(based upon Little et al.'s [Causal Bootstrapping](https://arxiv.org/abs/1910.09648) paper) to improve the
user-friendliness and performance of the technique when used for general causal de-confounding.

Causal Bootstrapping is a process for improving model robustness in the presence of causal factors (confounders) by
iteratively re-weighting samples within a dataset.

![(image: demonstration of causal bootstrapping on synthetic data)](bootstrapping.svg)

For reference on causality, I suggest reading the works of Judea Pearl. To my knowledge, neither of his books on the
topic (Causality, The Book of Why) are freely available outside of academia, however [Probabilistic and Causal Inference](https://ftp.cs.ucla.edu/pub/stat_ser/ACMBook-published-2022.pdf)
is, and may be a good place to get started.

## Usage and Development
- The project has been built with an assumed Python version of 3.12, however earlier and later versions may also work.
- I recommend `uv` and virtual environments for dependency management.
- Pull requests are welcome for new functionality or fixes!

Basic Usage:
```python
import causal_assistant as ca

# load your feature data
X, y, u = ...

# de-confound the data!
X_dc, y_dc = ca.bootstrap(causal_graph="""X;y;u;X->y;u->X;u->y;""", X=X, y=y, u=u)
```

## Attribution
Some code in this module is derived from the `causalBootstrapping` library.

If you use this library, please also cite the originating paper:
```
@article{little2019causal,
  title={Causal bootstrapping},
  author={Little, Max A and Badawy, Reham},
  journal={arXiv preprint arXiv:1910.09648},
  year={2019}
}
```