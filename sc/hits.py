"""A hit: a candidate the stability check calls plausible (within 0.1 eV/atom of the convex hull)
with a high predicted Tc. Each new hit is recorded in results/hits.json and opened as a GitHub
issue labelled "hit", so the owner gets a notification.

Thresholds:
  * any candidate with predicted Tc minus uncertainty (score) >= 77 K, above liquid nitrogen;
  * or, outside the cuprates, a score >= 40 K (rarer families matter more).
"""
import json
import os
import urllib.request

from pymatgen.core import Composition

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PATH = os.path.join(ROOT, "results", "hits.json")
REPO = "jasonbrelsford/superconductor-screen"
CUPRATE_MIN, OTHER_MIN = 77.0, 40.0
MAX_MATCH = 0.08   # cation-fraction distance between the computed structure and the candidate


def _is_cuprate(f):
    els = {str(e) for e in Composition(f).elements}
    return {"Cu", "O"} <= els


def find(board, checks):
    """New hits among `checks` (formula -> result), using scores from the leaderboard dataframe."""
    known = json.load(open(PATH)) if os.path.exists(PATH) else {}
    rows = board.set_index("formula")
    new = {}
    for f, r in checks.items():
        if r.get("status") != "plausible" or f in known or f not in rows.index:
            continue
        if r.get("match", 0.0) > MAX_MATCH:   # what was computed must really be this candidate
            continue
        row = rows.loc[f]
        score = float(row["score"])
        cup = _is_cuprate(f) or _is_cuprate(str(row["parent"]))
        if score >= CUPRATE_MIN or (not cup and score >= OTHER_MIN):
            new[f] = {"score": score, "pred_Tc_K": float(row["pred_Tc_K"]), "parent": str(row["parent"]),
                      "change": str(row["change"]), "e_above_hull_eV": r.get("e_above_hull_eV"),
                      "computed_as": r.get("computed_as"), "found": r.get("checked"), "by": r.get("by", "github")}
    if new:
        known.update(new)
        json.dump(known, open(PATH, "w"), indent=1, sort_keys=True)
    return new


def notify(new):
    """Open one GitHub issue per new hit. Uses GITHUB_TOKEN (Actions) or GITHUB_ADMIN_TOKEN (TOWER)."""
    token = os.environ.get("GITHUB_TOKEN") or os.environ.get("GITHUB_ADMIN_TOKEN")
    if not token or not new:
        return
    for f, h in new.items():
        body = (f"**{f}** passed the stability check and has a high predicted Tc.\n\n"
                f"- Predicted Tc: {h['pred_Tc_K']} K (score {h['score']} K after subtracting uncertainty)\n"
                f"- Energy above the CHGNet convex hull: {h['e_above_hull_eV']} eV/atom (computed as {h['computed_as']})\n"
                f"- Made from {h['parent']} by {h['change']}\n- Found {h['found']} by {h['by']}\n\n"
                "This is a prediction to examine, not a discovery: check the chemistry and the host structure "
                "before anyone tries to make it.")
        req = urllib.request.Request(f"https://api.github.com/repos/{REPO}/issues", method="POST",
                                     data=json.dumps({"title": f"Hit: {f} (predicted {h['pred_Tc_K']} K, plausible)",
                                                      "body": body, "labels": ["hit"]}).encode(),
                                     headers={"Authorization": f"Bearer {token}", "Accept": "application/vnd.github+json"})
        try:
            urllib.request.urlopen(req, timeout=30).read()
        except Exception as e:
            print("could not open issue for", f, e)
