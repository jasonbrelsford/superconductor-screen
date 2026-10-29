"""Generate new compositions by chemically sensible substitution into known superconductors.

For each parent (the best-Tc compound of each chemical system with Tc >= MIN_PARENT_TC):
  * full substitution: swap one element for a similar one (same group, neighbouring row;
    rare earths swap among each other);
  * partial doping: replace 10% or 20% of one element with a similar one.
Only elements the model has seen often enough are used, and anything already in the training
data is dropped, so every candidate is a composition the dataset does not contain.
"""
from collections import Counter

from pymatgen.core import Composition, Element

MIN_PARENT_TC = 20.0
MIN_ELEMENT_COUNT = 30          # element must appear in at least this many training compounds
DOPING = (0.1, 0.2)
EXCLUDE = {"Tc", "Pm", "Po", "At", "Rn", "Fr", "Ra", "Ac", "Th", "Pa", "U", "Np", "Pu",
           "He", "Ne", "Ar", "Kr", "Xe"}
ANIONS = {"O", "N", "F", "S", "Se", "Te", "Cl", "Br", "I", "H"}
RARE_EARTHS = ["Y", "La", "Ce", "Pr", "Nd", "Sm", "Eu", "Gd", "Tb", "Dy", "Ho", "Er", "Tm", "Yb", "Lu"]


def allowed_elements(train_df):
    count = Counter()
    for f in train_df["reduced"]:
        count.update(str(e) for e in Composition(f).elements)
    return {e for e, n in count.items() if n >= MIN_ELEMENT_COUNT and e not in EXCLUDE}


def similar(sym, allowed):
    if sym in RARE_EARTHS:
        return [r for r in RARE_EARTHS if r != sym and r in allowed]
    el = Element(sym)
    out = []
    for other in allowed:
        if other == sym:
            continue
        o = Element(other)
        same_group = o.group == el.group and abs(o.row - el.row) <= 1
        neighbour = o.row == el.row and abs(o.group - el.group) == 1 and el.group not in (1, 2, 17, 18)
        if same_group or neighbour:
            out.append(other)
    return out


def _fmt(amounts):
    comp = Composition({k: round(v, 4) for k, v in amounts.items() if v > 1e-9})
    return comp.reduced_formula


def generate(train_df):
    allowed = allowed_elements(train_df)
    known = set(train_df["reduced"])
    parents = (train_df[train_df["Tc"] >= MIN_PARENT_TC]
               .assign(sys=lambda d: d["reduced"].map(lambda f: "-".join(sorted(str(e) for e in Composition(f).elements))))
               .sort_values("Tc", ascending=False)
               .drop_duplicates("sys"))
    out = {}
    for _, row in parents.iterrows():
        amt = Composition(row["reduced"]).get_el_amt_dict()
        if not set(amt) <= allowed:
            continue
        for sym, a in amt.items():
            for sub in similar(sym, allowed):
                variants = []
                if sym not in ANIONS:   # swapping every oxygen (or F, S...) makes a different class of compound
                    full = dict(amt); full.pop(sym)
                    full[sub] = full.get(sub, 0) + a
                    variants.append(full)
                for x in DOPING:
                    d = dict(amt); d[sym] = a * (1 - x)
                    d[sub] = d.get(sub, 0) + a * x
                    variants.append(d)
                for v in variants:
                    f = _fmt(v)
                    if f not in known and f not in out:
                        x = 1.0 if not v.get(sym) else round(1 - v[sym] / a, 3)
                        out[f] = {"parent": row["reduced"], "parent_Tc": float(row["Tc"]),
                                  "change": f"{sym} partly -> {sub}" if v.get(sym) else f"{sym} -> {sub}",
                                  "swaps": [(sym, sub, x)]}
    return out


def _swap(amt, sym, sub, x):
    """Move fraction x (1.0 = all) of element sym to sub; anions are never fully swapped."""
    if x >= 1.0 and sym in ANIONS:
        return None
    d = dict(amt)
    a = d.pop(sym) if x >= 1.0 else d[sym]
    if x < 1.0:
        d[sym] = a * (1 - x)
    d[sub] = d.get(sub, 0) + a * x
    return d


def explore(train_df, seed, per_parent=40):
    """Today's random batch: two substitutions at once, drawn with a date-based seed so every
    daily run looks somewhere new (and any run can be reproduced from its date)."""
    import random
    rng = random.Random(seed)
    allowed = allowed_elements(train_df)
    known = set(train_df["reduced"])
    parents = (train_df[train_df["Tc"] >= MIN_PARENT_TC]
               .assign(sys=lambda d: d["reduced"].map(lambda f: "-".join(sorted(str(e) for e in Composition(f).elements))))
               .sort_values("Tc", ascending=False)
               .drop_duplicates("sys"))
    out = {}
    for _, row in parents.iterrows():
        amt = Composition(row["reduced"]).get_el_amt_dict()
        if not set(amt) <= allowed or len(amt) < 2:
            continue
        options = [(s, t) for s in amt for t in similar(s, allowed)]
        if len(options) < 2:
            continue
        for _ in range(per_parent):
            (s1, t1), (s2, t2) = rng.sample(options, 2)
            if s1 == s2:
                continue
            x1, x2 = rng.choice((1.0, 0.1, 0.2, 0.3)), rng.choice((1.0, 0.1, 0.2, 0.3))
            d = _swap(amt, s1, t1, x1)
            d = d and _swap(d, s2, t2, x2) if d and s2 in d else None
            if not d:
                continue
            f = _fmt(d)
            if f not in known and f not in out:
                out[f] = {"parent": row["reduced"], "parent_Tc": float(row["Tc"]),
                          "change": f"{s1} {'->' if x1 >= 1 else 'partly ->'} {t1}; {s2} {'->' if x2 >= 1 else 'partly ->'} {t2}",
                          "swaps": [(s1, t1, x1), (s2, t2, x2)]}
    return out
