"""A hit: a candidate the stability check calls plausible (within 0.1 eV/atom of the convex hull)
with a high predicted Tc. Each new hit is recorded in results/hits.json and opened as a GitHub
issue labelled "hit", so the owner gets a notification.

Thresholds:
  * a cuprate whose score (predicted Tc minus uncertainty) is >= 77 K, above liquid nitrogen;
  * or, outside the cuprates, a score >= 40 K (rarer families matter more);
  * and in both cases the score must beat the measured Tc of the parent and the best measured Tc
    of any known compound with the same main elements.
"""
import json
import os
import urllib.request

from pymatgen.core import Composition, Element

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PATH = os.path.join(ROOT, "results", "hits.json")
REPO = "jasonbrelsford/superconductor-screen"
CUPRATE_MIN, OTHER_MIN = 77.0, 40.0
MAX_MATCH = 0.08   # cation-fraction distance between the computed structure and the candidate


ANION_ELS = {"O", "F", "N", "S", "Cl", "Br", "I", "Se", "Te", "P", "As"}


def charge_ok(f):
    """False for formulas no assignment of usual oxidation states can balance (typos in the source
    data such as Ba23...O7.6). Only judged for ionic compounds; oxygen written as exactly 1 is the
    database's placeholder for unknown oxygen content and is left out."""
    amt = Composition(f).get_el_amt_dict()
    if not ANION_ELS & set(amt):
        return True
    if amt.get("O") == 1.0:
        amt.pop("O")
        if not ANION_ELS & set(amt):
            return True
    lo = hi = neg = 0.0
    for e, a in amt.items():
        el = Element(e)
        st = set(el.common_oxidation_states) | set(el.icsd_oxidation_states or ()) or set(el.oxidation_states) or {0}
        lo += min(st) * a
        hi += max(st) * a
        neg += abs(min(st)) * a if e in ANION_ELS else 0
    tol = 0.1 * max(neg, 1.0)
    return lo <= tol and hi >= -tol


def high_pressure(f):
    """Hydrogen-rich compounds (H3S, LaH10 and relatives) superconduct only under megabar pressure;
    the ambient-pressure stability check says nothing about them."""
    amt = Composition(f).get_el_amt_dict()
    return amt.get("H", 0) / sum(amt.values()) >= 0.3


def family(f):
    """The main elements of a formula: anions plus cations holding at least 8% of the cation sites."""
    amt = Composition(f).get_el_amt_dict()
    cat = {e: a for e, a in amt.items() if e not in ANION_ELS}
    tot = sum(cat.values()) or 1.0
    return frozenset(e for e in amt if e in ANION_ELS or cat[e] / tot >= 0.08)


RARE = {"Sc", "Y", "La", "Ce", "Pr", "Nd", "Pm", "Sm", "Eu", "Gd", "Tb", "Dy", "Ho", "Er", "Tm", "Yb", "Lu"}
NEAR = 0.25   # cation-fraction distance within which a known compound counts as the same family


def _mapped(f):
    """Cation fractions with rare earths pooled as RE and Sr/Ba pooled as AE (they share sites), and
    the anion set; used to find the known compounds a candidate is a variant of."""
    amt = Composition(f).get_el_amt_dict()
    cat = {}
    for e, a in amt.items():
        if e in ANION_ELS:
            continue
        k = "RE" if e in RARE else "AE" if e in ("Sr", "Ba") else e
        cat[k] = cat.get(k, 0.0) + a
    tot = sum(cat.values()) or 1.0
    cat = {k: a / tot for k, a in cat.items()}
    return cat, frozenset(e for e in amt if e in ANION_ELS and amt[e] >= 0.25 * max(
        [amt[x] for x in amt if x in ANION_ELS] or [0]))


def records(df):
    """Index of the training data for record_for(): known compounds grouped by their major cations."""
    full, minus1 = {}, {}
    for f, tc in zip(df["reduced"], df["Tc"]):
        try:
            cat, an = _mapped(f)
        except Exception:
            continue
        major = frozenset(k for k, x in cat.items() if x >= 0.08)
        item = (cat, an, float(tc))
        full.setdefault(major, []).append(item)
        for e in major:
            minus1.setdefault(major - {e}, []).append(item)
    return {"full": full, "minus1": minus1}


def record_for(f, recs):
    """Best measured Tc among known compounds that this formula is a close variant of: same main
    anions, cation fractions within NEAR (rare earths and Sr/Ba pooled), at most one major cation
    more or fewer. 0.0 if none."""
    if not recs:
        return 0.0
    cat, an = _mapped(f)
    major = frozenset(k for k, x in cat.items() if x >= 0.08)
    pool = list(recs["full"].get(major, [])) + list(recs["minus1"].get(major, []))
    for e in major:
        pool += recs["full"].get(major - {e}, [])
    best = 0.0
    for kc, kan, tc in pool:
        if kan != an or tc <= best:
            continue
        if sum(abs(cat.get(k, 0) - kc.get(k, 0)) for k in set(cat) | set(kc)) <= NEAR:
            best = tc
    return best


def _is_cuprate(f):
    els = {str(e) for e in Composition(f).elements}
    return {"Cu", "O"} <= els


def find(board, checks, recs=None):
    """New hits among `checks` (formula -> result), using scores from the leaderboard dataframe.
    `recs` (from records()) sets the bar: a hit must beat the best measured Tc of its family, so a
    stable copy of a known record holder reached by another route is not news."""
    recs = recs or {}
    known = json.load(open(PATH)) if os.path.exists(PATH) else {}
    rows = board.set_index("formula")
    new = {}
    for f, r in checks.items():
        if r.get("status") != "plausible" or f in known or f not in rows.index:
            continue
        if r.get("match", 0.0) > MAX_MATCH:   # what was computed must really be this candidate
            continue
        if not charge_ok(f) or high_pressure(f):
            continue
        row = rows.loc[f]
        new_els = {str(e) for e in Composition(f).elements} - {str(e) for e in Composition(str(row["parent"])).elements}
        if new_els and not new_els & {str(e) for e in Composition(r.get("computed_as", "")).elements}:
            continue   # every new element was too dilute to place: what was computed is the parent
        score = float(row["score"])
        cup = _is_cuprate(f) or _is_cuprate(str(row["parent"]))
        beats = score > max(float(row["parent_Tc"]), record_for(f, recs))
        if beats and ((cup and score >= CUPRATE_MIN) or (not cup and score >= OTHER_MIN)):
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
    api = f"https://api.github.com/repos/{REPO}/issues"
    hdr = {"Authorization": f"Bearer {token}", "Accept": "application/vnd.github+json"}
    earlier = {}   # formula -> issue number, so a hit found again reopens its old issue
    try:
        for page in range(1, 6):
            got = json.load(urllib.request.urlopen(urllib.request.Request(
                f"{api}?labels=hit&state=all&per_page=100&page={page}", headers=hdr), timeout=30))
            for i in got:
                earlier[i["title"].removeprefix("Hit: ").split(" (")[0]] = i["number"]
            if len(got) < 100:
                break
    except Exception as e:
        print("could not list earlier hit issues", e)
    for f, h in new.items():
        body = (f"**{f}** passed the stability check and has a high predicted Tc.\n\n"
                f"- Predicted Tc: {h['pred_Tc_K']} K (score {h['score']} K after subtracting uncertainty)\n"
                f"- Energy above the CHGNet convex hull: {h['e_above_hull_eV']} eV/atom (computed as {h['computed_as']})\n"
                f"- Made from {h['parent']} by {h['change']}\n- Found {h['found']} by {h['by']}\n\n"
                "This is a prediction to examine, not a discovery: check the chemistry and the host structure "
                "before anyone tries to make it.")
        if f in earlier:   # reopen with the new numbers instead of a duplicate issue
            n = earlier[f]
            reqs = [urllib.request.Request(f"{api}/{n}", method="PATCH", headers=hdr,
                                           data=json.dumps({"state": "open"}).encode()),
                    urllib.request.Request(f"{api}/{n}/comments", method="POST", headers=hdr,
                                           data=json.dumps({"body": "Passed again:\n\n" + body}).encode())]
            try:
                for r in reqs:
                    urllib.request.urlopen(r, timeout=30).read()
            except Exception as e:
                print("could not reopen issue for", f, e)
            continue
        req = urllib.request.Request(api, method="POST",
                                     data=json.dumps({"title": f"Hit: {f} (predicted {h['pred_Tc_K']} K, plausible)",
                                                      "body": body, "labels": ["hit"]}).encode(),
                                     headers=hdr)
        try:
            urllib.request.urlopen(req, timeout=30).read()
        except Exception as e:
            print("could not open issue for", f, e)
