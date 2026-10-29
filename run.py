"""Superconductor screen: train, generate, rank, check stability, publish.

    python run.py            # writes results/<date>/*.csv, results/leaderboard.csv, results/latest.json, site/index.html
"""
import datetime as dt
import html
import json
import os

import numpy as np
import pandas as pd
from pymatgen.core import Composition

from sc import candidates, data, model, stability
from sc.features import featurize

TOP = 50
PER_PARENT = 2          # at most two candidates built from the same parent compound per list
STABILITY_CHECKS = 150
LEADERBOARD_MIN_SCORE = 80.0   # keep candidates scoring at least this (K) on the running leaderboard
ROOT = os.path.dirname(os.path.abspath(__file__))
COLS = ["formula", "first_seen", "pred_Tc_K", "uncertainty_K", "score", "vs_parent_K", "parent", "parent_Tc", "change",
        "stability", "e_above_hull_eV"]


def is_cuprate(formula):
    el = {str(e) for e in Composition(formula).elements}
    return {"Cu", "O"} <= el


def pick(res, sort_col, mask=None):
    d = res if mask is None else res[mask]
    d = d.sort_values(sort_col, ascending=False)
    return d.drop_duplicates("system").groupby("parent", sort=False).head(PER_PARENT).head(TOP).copy()


def main():
    today = dt.date.today().isoformat()
    df = data.load()
    elements = sorted({str(e) for f in df["reduced"] for e in Composition(f).elements})
    print(f"training compounds: {len(df)}, elements: {len(elements)}")
    models, metrics = model.train(df, elements)
    print("metrics:", metrics)

    cands = candidates.generate(df)
    seed = int(dt.datetime.now(dt.timezone.utc).strftime("%Y%m%d%H"))   # new batch every run, reproducible from the hour
    fresh = candidates.explore(df, seed=seed)
    print(f"single substitutions: {len(cands)}, today's random double substitutions: {len(fresh)}")
    cands.update({f: v for f, v in fresh.items() if f not in cands})
    forms = list(cands)
    X = np.vstack([featurize(f, elements) for f in forms])
    mean, std = model.predict(models, X)
    res = pd.DataFrame({"formula": forms, "pred_Tc_K": np.round(mean.astype(float), 1), "uncertainty_K": np.round(std.astype(float), 1)})
    res["score"] = (res["pred_Tc_K"] - res["uncertainty_K"]).round(1)   # conservative: mean minus spread
    for k in ("parent", "parent_Tc", "change"):
        res[k] = [cands[f][k] for f in forms]
    res["system"] = res["formula"].map(model.chemsys)
    # Only element combinations nobody has measured yet: a 1% tweak of a known compound tells us nothing.
    known_systems = set(df["reduced"].map(model.chemsys))
    res = res[~res["system"].isin(known_systems)].copy()
    print(f"in new chemical systems today: {len(res)}")
    # Running leaderboard: every strong candidate any run has found, with the date it first appeared.
    board_path = os.path.join(ROOT, "results", "leaderboard.csv")
    res["first_seen"] = today
    if os.path.exists(board_path):
        old = pd.read_csv(board_path)
        res = pd.concat([old, res[~res["formula"].isin(set(old["formula"]))]], ignore_index=True)
    for c in ("pred_Tc_K", "uncertainty_K", "score", "parent_Tc"):
        res[c] = res[c].astype(float).round(1)
    os.makedirs(os.path.dirname(board_path), exist_ok=True)
    keep = res[res["score"] >= LEADERBOARD_MIN_SCORE]   # the file keeps strong ones; today's lists use everything
    keep.drop(columns=[c for c in ("vs_parent_K", "gain_score", "cuprate") if c in keep]).to_csv(board_path, index=False, float_format="%.4g")
    res["vs_parent_K"] = (res["pred_Tc_K"] - res["parent_Tc"]).round(1)
    res["gain_score"] = res["vs_parent_K"] - res["uncertainty_K"]
    res["cuprate"] = res["formula"].map(is_cuprate) | res["parent"].map(is_cuprate)
    print(f"leaderboard size: {len(res)}")

    lists = {
        "highest": ("Highest predicted Tc", "New element combinations with the highest conservative score. Mostly cuprates, as in real high-Tc research.",
                    pick(res, "score")),
        "gain": ("Biggest predicted gain over the parent", "Substitutions the model expects to raise Tc above the compound they came from, after subtracting uncertainty.",
                 pick(res, "gain_score", res["gain_score"] > 0)),
        "noncuprate": ("Best outside the cuprates", "No copper-oxide layers: iron pnictides, borides, hydrides, nickelates and others.",
                       pick(res, "score", ~res["cuprate"])),
    }

    to_check = []
    for _, _, t in lists.values():
        to_check += [f for f in t["formula"] if f not in to_check]
    stab = stability.check(to_check[:STABILITY_CHECKS])
    for _, _, t in lists.values():
        if stab is None:
            t["stability"] = "not checked yet"
            t["e_above_hull_eV"] = ""
        else:
            t["stability"] = [stab.get(f, {}).get("status", "not checked") for f in t["formula"]]
            t["e_above_hull_eV"] = [stab.get(f, {}).get("e_above_hull", "") for f in t["formula"]]

    outdir = os.path.join(ROOT, "results", today)
    os.makedirs(outdir, exist_ok=True)
    for key, (_, _, t) in lists.items():
        t[COLS].to_csv(os.path.join(outdir, f"{key}.csv"), index=False)
    summary = {"date": today, "metrics": metrics, "n_candidates_new_systems": int(len(res)),
               "stability_checked": stab is not None,
               "lists": {k: t[COLS].head(10).to_dict("records") for k, (_, _, t) in lists.items()}}
    with open(os.path.join(ROOT, "results", "latest.json"), "w") as f:
        json.dump(summary, f, indent=1, default=str)
    write_site(today, metrics, len(res), lists, stab is not None)


def table(t):
    e = html.escape
    head = "".join(f"<th>{e(c)}</th>" for c in COLS)
    rows = "\n".join("<tr>" + "".join(f"<td>{e(str(r[c]))}</td>" for c in COLS) + "</tr>" for _, r in t.iterrows())
    return f'<div class="wrap"><table><thead><tr>{head}</tr></thead><tbody>{rows}</tbody></table></div>'


def write_site(today, metrics, n, lists, checked):
    e = html.escape
    g, rcv = metrics["grouped_cv"], metrics["random_cv"]
    stab_note = ("Candidates were checked against the Materials Project: 'near hull' means a computed structure of "
                 "that composition is thermodynamically plausible; 'not in Materials Project' means nobody has computed "
                 "its structure, so stability is unknown." if checked else
                 "Stability has not been checked yet (the Materials Project key is not set up).")
    sections = "".join(f"<h2>{e(title)}</h2><p class='mute'>{e(desc)}</p>{table(t)}" for title, desc, t in lists.values())
    page = f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Superconductor Screen</title>
<style>
:root{{--bg:#fff;--fg:#1a1a1a;--mute:#5f6368;--line:#e2e2e2;--accent:#0b6e4f}}
@media (prefers-color-scheme:dark){{:root{{--bg:#121212;--fg:#ececec;--mute:#a0a0a0;--line:#2c2c2c;--accent:#4fd1a5}}}}
body{{background:var(--bg);color:var(--fg);font:16px/1.55 system-ui,sans-serif;margin:0 auto;padding:24px 16px;max-width:1100px}}
h1{{font-size:1.6rem;margin:0 0 4px}} h2{{font-size:1.2rem;margin:28px 0 4px}} .mute{{color:var(--mute)}} a{{color:var(--accent)}}
.wrap{{overflow-x:auto;border:1px solid var(--line);border-radius:8px;max-height:520px}}
table{{border-collapse:collapse;width:100%;font-size:14px}} th,td{{padding:6px 10px;border-bottom:1px solid var(--line);text-align:left;white-space:nowrap}}
th{{position:sticky;top:0;background:var(--bg)}}
.card{{border:1px solid var(--line);border-radius:8px;padding:12px 16px;margin:16px 0}}
</style></head><body>
<h1>Superconductor Screen</h1>
<p class="mute">Run {e(today)} &middot; {n:,} candidates on the running leaderboard (predicted Tc minus uncertainty of 80 K or more), new random substitutions added every day &middot;
<a href="https://github.com/jasonbrelsford/superconductor-screen">method, code and every past run</a></p>
<div class="card"><strong>What this is.</strong> A machine-learning model trained on {metrics['n_compounds']:,} known
superconductors predicts the critical temperature (Tc) of new compositions made by swapping chemically similar elements
into known superconductors. These are <em>predictions to test</em>, not discoveries: a material counts only when a lab
makes it and measures zero resistance and the Meissner effect. {e(stab_note)}</div>
<div class="card"><strong>How good is the model?</strong> On chemical systems it never saw (the honest test), mean error
is {g['mae_K']} K (R&sup2; {g['r2']}); for Tc above 30 K it is {g['mae_K_on_Tc_30K_plus']} K. On a random split it looks
better ({rcv['mae_K']} K, R&sup2; {rcv['r2']}) because similar compounds leak between training and test.
<em>score</em> = predicted Tc minus the spread of five models, so uncertain guesses rank lower; <em>vs_parent_K</em> is
the predicted change from the known compound it came from.</div>
{sections}
<p class="mute">Data: SuperCon-derived dataset, Hamidieh (2018), UCI Machine Learning Repository #464. Stability:
Materials Project. Predictions only; no claim is made that any listed compound is a superconductor.</p>
</body></html>"""
    os.makedirs(os.path.join(ROOT, "site"), exist_ok=True)
    with open(os.path.join(ROOT, "site", "index.html"), "w", encoding="utf-8") as f:
        f.write(page)


if __name__ == "__main__":
    main()
