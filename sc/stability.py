"""Check top candidates against the Materials Project (needs MP_API_KEY).

A Tc prediction says nothing about whether the compound can exist. For each candidate we ask
the Materials Project whether that exact composition (reduced formula) has been computed:
  * known and on or near the convex hull (e_above_hull <= 0.1 eV/atom): plausible to make;
  * known but well above the hull: likely to decompose;
  * not in the database: no computed crystal structure yet, so stability is unknown and would
    need a DFT or ML-potential calculation. Doped (fractional) formulas usually land here.
Without a key this step is skipped and the report says so.
"""
import os


def check(formulas):
    key = os.environ.get("MP_API_KEY", "").strip()
    if not key:
        return None
    from mp_api.client import MPRester
    out = {}
    with MPRester(key) as mpr:
        for f in formulas:
            try:
                docs = mpr.materials.summary.search(formula=f, fields=["material_id", "energy_above_hull", "formula_pretty"])
            except Exception as e:  # fractional formulas are rejected by the API
                out[f] = {"status": "not in Materials Project", "detail": type(e).__name__}
                continue
            if not docs:
                out[f] = {"status": "not in Materials Project"}
                continue
            best = min(docs, key=lambda d: d.energy_above_hull if d.energy_above_hull is not None else 9)
            e = best.energy_above_hull
            out[f] = {"status": "near hull" if e is not None and e <= 0.1 else "above hull",
                      "e_above_hull": None if e is None else round(float(e), 3),
                      "mp_id": str(best.material_id)}
    return out
