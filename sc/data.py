"""Load the public SuperCon-derived dataset (Hamidieh 2018, UCI ML Repository #464)."""
import io
import os
import urllib.request
import zipfile

import pandas as pd
from pymatgen.core import Composition

URL = "https://archive.ics.uci.edu/static/public/464/superconductivty+data.zip"
CACHE = os.path.join(os.path.dirname(__file__), "..", "data", "sc.zip")


def load():
    if not os.path.exists(CACHE):
        os.makedirs(os.path.dirname(CACHE), exist_ok=True)
        urllib.request.urlretrieve(URL, CACHE)
    with zipfile.ZipFile(CACHE) as z:
        df = pd.read_csv(io.BytesIO(z.read("unique_m.csv")))
    df = df[["material", "critical_temp"]].rename(columns={"material": "formula", "critical_temp": "Tc"})
    keep = []
    for f in df["formula"]:
        try:
            c = Composition(f)
            keep.append(c.reduced_formula if c.num_atoms > 0 else None)
        except Exception:
            keep.append(None)
    df["reduced"] = keep
    df = df.dropna(subset=["reduced"])
    # The same compound measured more than once: keep the median Tc.
    return (df.groupby("reduced", as_index=False)
              .agg(formula=("formula", "first"), Tc=("Tc", "median")))
