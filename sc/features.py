"""Turn a chemical formula into a fixed-length numeric feature vector.

Two parts:
  * element fractions for every element seen in training (what the composition is), and
  * fraction-weighted mean, spread, min and max of a handful of element properties
    (what kind of atoms those are), in the spirit of the Magpie descriptors.
"""
import math
from functools import lru_cache

import numpy as np
from pymatgen.core import Composition, Element

PROPS = ["Z", "atomic_mass", "X", "atomic_radius", "mendeleev_no", "row", "group",
         "melting_point", "molar_volume", "thermal_conductivity", "electrical_resistivity",
         "nvalence"]


def _num(v):
    try:
        f = float(v)
    except (TypeError, ValueError):
        try:
            f = float(str(v).split()[0])  # values like "1234 K"
        except (TypeError, ValueError, IndexError):
            return math.nan
    return f if math.isfinite(f) else math.nan


@lru_cache(maxsize=None)
def element_props(sym):
    el = Element(sym)
    out = []
    for p in PROPS:
        if p == "nvalence":
            try:
                out.append(float(el.valence[1]))
            except Exception:
                out.append(math.nan)
        else:
            out.append(_num(getattr(el, p, None)))
    return np.array(out, dtype=float)


def featurize(formula, elements):
    """Feature vector for one formula. `elements` fixes the order of the fraction block."""
    comp = Composition(formula).fractional_composition
    fr = {str(e): a for e, a in comp.get_el_amt_dict().items()}
    frac = np.array([fr.get(e, 0.0) for e in elements])
    syms, w = list(fr), np.array(list(fr.values()))
    P = np.vstack([element_props(s) for s in syms])  # atoms x props
    stats = []
    for j in range(P.shape[1]):
        col = P[:, j]
        ok = ~np.isnan(col)
        if not ok.any():
            stats += [math.nan] * 4
            continue
        cw, cv = w[ok] / w[ok].sum(), col[ok]
        mean = float((cw * cv).sum())
        std = float(math.sqrt((cw * (cv - mean) ** 2).sum()))
        stats += [mean, std, float(cv.min()), float(cv.max())]
    return np.concatenate([frac, np.array(stats), [len(syms)]])


def feature_names(elements):
    names = [f"frac_{e}" for e in elements]
    for p in PROPS:
        names += [f"{p}_mean", f"{p}_std", f"{p}_min", f"{p}_max"]
    return names + ["n_elements"]
