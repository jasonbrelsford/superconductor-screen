"""Estimate whether a candidate can exist, using a real crystal structure and an ML potential.

For a candidate made by fully swapping elements in a parent compound:
  1. find a host crystal structure for the parent in the Materials Project (same elements, or the
     same elements without minor dopants; the stable phase whose composition is closest);
  2. apply the swaps to that structure (e.g. every Sr site becomes Ba);
  3. relax it with CHGNet, a universal machine-learned interatomic potential trained on
     Materials Project calculations;
  4. relax every stable competing phase of the new chemical system with the same model, and
     compute the candidate's energy above that convex hull. All energies come from one model, so
     no cross-method corrections are needed.
A result at or below about 0.1 eV/atom means "plausibly makeable"; far above means it would
probably decompose into the competing phases. This is an estimate: CHGNet errors are typically a
few tens of meV/atom, the host structure may not be the one the real compound would adopt, and
competing phases with very large cells are skipped (which can make a candidate look more stable).
Doped (partial) substitutions use the smallest supercell that hits the doping level within 5%,
with a few random dopant arrangements; the lowest-energy one counts. Real dopants may order
differently, so doped results are rougher than full substitutions.

The computed formula follows the host structure, so it can differ from the screen's formula,
especially in oxygen content, which the training data often records loosely.
"""
import itertools
import json
import os
import time

from pymatgen.analysis.phase_diagram import PDEntry, PhaseDiagram
from pymatgen.core import Composition, Element

MAX_SITES = 80
MAX_DOPED_SITES = 160   # largest supercell used for a doped candidate
CONFIGS = 3             # random dopant arrangements tried; the lowest-energy one counts
STEPS = 300
FMAX = 0.1


ANIONS = {"O", "N", "F", "S", "Se", "Te", "Cl", "Br", "I", "H", "B", "C", "P", "As"}
MINOR = 0.08        # cations below this share of the cation sites are treated as dopants
MAX_HOST_DISTANCE = 0.25
TRACE = 0.03       # dopants below 3% of a site are left out of the calculation (and listed as omitted)
RARE_EARTHS = ["Y", "La", "Ce", "Pr", "Nd", "Sm", "Eu", "Gd", "Tb", "Dy", "Ho", "Er", "Tm", "Yb", "Lu"]


def _cations(comp):
    """Cation-site fractions only: anion counts in the training data are often loose."""
    c = Composition(comp)
    cat = {str(e): a for e, a in c.get_el_amt_dict().items() if str(e) not in ANIONS}
    tot = sum(cat.values()) or 1.0
    return {e: a / tot for e, a in cat.items()}


def _frac_distance(a, b):
    ca, cb = _cations(a), _cations(b)
    return sum(abs(ca.get(e, 0) - cb.get(e, 0)) for e in set(ca) | set(cb))


class Checker:
    def __init__(self, key, cache_path, budget_s):
        from chgnet.model import StructOptimizer
        from mp_api.client import MPRester
        self.mpr = MPRester(key)
        self.opt = StructOptimizer()
        self.cache_path = cache_path
        self.cache = json.load(open(cache_path)) if os.path.exists(cache_path) else {}
        self.deadline = time.time() + budget_s

    def close(self):
        json.dump(self.cache, open(self.cache_path, "w"), indent=0, sort_keys=True)
        self.mpr.__exit__(None, None, None)

    def time_left(self):
        return self.deadline - time.time()

    def _relax(self, structure):
        r = self.opt.relax(structure, fmax=FMAX, steps=STEPS, verbose=False)
        return r["final_structure"], float(r["trajectory"].energies[-1])

    def _stable_in(self, chemsys_list):
        docs = self.mpr.materials.summary.search(chemsys=chemsys_list, is_stable=True,
                                                 fields=["material_id", "structure", "nsites"])
        return docs

    def major_elements(self, parent):
        els = {str(e) for e in Composition(parent).elements}
        cat = _cations(parent)
        return {e for e in els if e in ANIONS or cat.get(e, 0) >= MINOR}

    def host(self, parent):
        """Closest stable-ish MP structure for the parent, or (None, reason). Tries the parent's major
        elements first; if Materials Project has nothing there, drops the least abundant cations one
        or two at a time (the placement step then puts them back as substitutions)."""
        major = self.major_elements(parent)
        cat = _cations(parent)
        cations = sorted((e for e in major if e not in ANIONS), key=lambda e: cat.get(e, 0))
        tries = [major]
        for n in (1, 2):
            for drop in itertools.combinations(cations[:4], n):
                if len(cations) - n >= 2:
                    tries.append(major - set(drop))
        closest = None
        for els in tries:
            docs = self.mpr.materials.summary.search(chemsys="-".join(sorted(els)),
                                                     fields=["material_id", "structure", "energy_above_hull", "nsites"])
            docs = [d for d in docs if d.nsites <= MAX_SITES and d.energy_above_hull is not None and d.energy_above_hull <= 0.1]
            if not docs:
                continue
            best = min(docs, key=lambda d: (_frac_distance(d.structure.composition, parent), d.energy_above_hull))
            dist = _frac_distance(best.structure.composition, parent)
            if dist <= MAX_HOST_DISTANCE:
                return best, ""
            if closest is None or dist < closest[0]:
                closest = (dist, best)
        if closest:
            return None, f"closest host too different ({closest[1].structure.composition.reduced_formula})"
        return None, "no host structure in Materials Project"

    def competitor_energies(self, elements):
        """CHGNet energy per atom of every stable MP phase in the chemical system (cached by mp-id)."""
        subsystems = ["-".join(sorted(c)) for n in range(1, len(elements) + 1)
                      for c in itertools.combinations(sorted(elements), n)]
        entries = []
        for d in self._stable_in(subsystems):
            mid = str(d.material_id)
            if d.nsites > MAX_SITES:
                continue
            if mid not in self.cache:
                if self.time_left() < 60:
                    raise TimeoutError
                s, e = self._relax(d.structure)
                self.cache[mid] = {"formula": s.composition.formula, "e_total": e}
            c = self.cache[mid]
            entries.append(PDEntry(Composition(c["formula"]), c["e_total"]))
        return entries

    # ---- placing a candidate's atoms on the host structure ------------------------------------

    @staticmethod
    def _similarity(a, b):
        """0 = interchangeable on a site, larger = less likely to share a site."""
        if a == b:
            return 0.0
        ea, eb = Element(a), Element(b)
        rare = set(RARE_EARTHS)
        if (a in rare and b in rare) or ea.group == eb.group:
            return 0.0
        if abs(ea.group - eb.group) == 1 and ea.row == eb.row:
            return 0.3
        try:
            ra, rb = ea.average_ionic_radius, eb.average_ionic_radius
            if ra and rb:
                return min(1.0, abs(float(ra) - float(rb)) / float(max(ra, rb)))
        except Exception:
            pass
        return 1.0

    def plan(self, host_comp, formula):
        """Which host sites get which new elements, as fractions of that site, read from the
        candidate's formula. Cations and anions are matched separately (anion totals in the
        training data are loose, so only their ratios are used). Returns (plan, omitted)."""
        host = Composition(host_comp).get_el_amt_dict()
        cand = Composition(formula).get_el_amt_dict()
        plan, omitted = {}, []
        for group in (lambda e: e not in ANIONS, lambda e: e in ANIONS):
            h = {e: a for e, a in host.items() if group(e)}
            c = {e: a for e, a in cand.items() if group(e)}
            if not h or not c:
                continue
            scale = sum(h.values()) / sum(c.values())
            c = {e: a * scale for e, a in c.items()}
            deficit = {e: max(0.0, h[e] - c.get(e, 0.0)) for e in h}
            extras = {e: a for e, a in c.items() if e not in h}
            for e in h:   # an over-full host element spills onto another site
                if c.get(e, 0.0) - h[e] > 0.02 * h[e]:
                    extras[e] = c[e] - h[e]
            for m, a in sorted(extras.items(), key=lambda kv: -kv[1]):
                options = [(abs(deficit[s] - a) / max(a, 1e-9) + 2 * self._similarity(m, s), s)
                           for s in h if s != m and deficit[s] > 0.25 * a]
                if not options:
                    omitted.append(m)
                    continue
                _, site = min(options)
                take = min(a, deficit[site])
                deficit[site] -= take
                plan.setdefault(site, {})[m] = plan.get(site, {}).get(m, 0.0) + take / h[site]
        for site in list(plan):
            for m in list(plan[site]):
                if plan[site][m] < TRACE:
                    omitted.append(m)
                    del plan[site][m]
            if not plan[site]:
                del plan[site]
        return plan, omitted

    def _supercell_for(self, s, plan):
        """Smallest supercell (<= MAX_DOPED_SITES) where every fraction in the plan is hit within
        0.05 by a whole number of sites (full replacements always fit)."""
        counts = {}
        for site in s:
            counts[site.specie.symbol] = counts.get(site.specie.symbol, 0) + 1
        best = None
        mats = [(1, 1, 1), (2, 1, 1), (1, 2, 1), (1, 1, 2), (2, 2, 1), (2, 1, 2), (1, 2, 2), (3, 1, 1),
                (1, 3, 1), (1, 1, 3), (2, 2, 2), (3, 2, 1), (3, 1, 2), (2, 3, 1), (4, 1, 1), (1, 4, 1),
                (1, 1, 4), (3, 3, 1), (3, 1, 3), (1, 3, 3), (4, 2, 1), (2, 2, 3), (3, 2, 2), (2, 3, 2), (4, 2, 2), (3, 3, 2)]
        for m in mats:
            mult = m[0] * m[1] * m[2]
            n = len(s) * mult
            if n > MAX_DOPED_SITES:
                continue
            ok = True
            for site_el, subs in plan.items():
                sites = counts.get(site_el, 0) * mult
                if sum(round(x * sites) for x in subs.values()) > sites:
                    ok = False
                for x in subs.values():
                    k = round(x * sites)
                    if k < 1 or (x < 0.999 and abs(k / sites - x) > 0.05):
                        ok = False
            if ok and (best is None or n < best[0]):
                best = (n, m)
        return None if best is None else best[1]

    def check(self, formula, parent, swaps=None):
        h, why = self.host(parent)
        if h is None:
            return {"status": why}
        base = h.structure.copy()
        plan, omitted = self.plan(base.composition, formula)
        if not plan:
            return {"status": "candidate does not differ from host", "host": str(h.material_id)}
        full = {Element(site): Element(next(iter(subs))) for site, subs in plan.items()
                if len(subs) == 1 and next(iter(subs.values())) >= 0.97}
        partial = {site: subs for site, subs in plan.items() if Element(site) not in full}
        if full:
            base.replace_species(full)
        configs = [base]
        cell = (1, 1, 1)
        if partial:
            cell = self._supercell_for(base, partial)
            while cell is None:   # drop the smallest dopant until the rest fits a cell we can afford
                site, m = min(((site, m) for site, subs in partial.items() for m in subs),
                              key=lambda sm: partial[sm[0]][sm[1]])
                omitted.append(m)
                del partial[site][m]
                if not partial[site]:
                    del partial[site]
                if not partial:
                    break
                cell = self._supercell_for(base, partial)
            if cell is None:
                cell = (1, 1, 1)
            if not partial and not full:
                return {"status": "doping level needs too large a cell", "host": str(h.material_id)}
            big = base * cell
            import random
            rng = random.Random(formula)
            configs = []
            for _ in range(CONFIGS if partial else 1):
                c = big.copy()
                for site_el, subs in partial.items():
                    idx = [i for i, st in enumerate(c) if st.specie.symbol == site_el]
                    rng.shuffle(idx)
                    pos = 0
                    for new, x in subs.items():
                        k = round(x * len(idx))
                        for i in idx[pos:pos + k]:
                            c.replace(i, new)
                        pos += k
                configs.append(c)
        best = None
        for c in configs:
            if self.time_left() < 120:
                raise TimeoutError
            relaxed, e = self._relax(c)
            if best is None or e / len(relaxed) < best[1] / len(best[0]):
                best = (relaxed, e)
        relaxed, e = best
        els = [str(x) for x in relaxed.composition.elements]
        comps = self.competitor_energies(els)
        have = {str(x) for en in comps for x in en.composition.elements if en.composition.is_element}
        if set(els) - have:
            return {"status": "missing elemental reference", "host": str(h.material_id)}
        pd = PhaseDiagram(comps)
        _, ehull = pd.get_decomp_and_e_above_hull(PDEntry(relaxed.composition, e), allow_negative=True)
        ehull = round(float(ehull), 3)
        match = round(_frac_distance(relaxed.composition, formula), 3)
        out = {"status": "plausible" if ehull <= 0.1 else "likely unstable",
               "e_above_hull_eV": ehull, "computed_as": relaxed.composition.reduced_formula,
               "cell_atoms": len(relaxed), "host": str(h.material_id), "match": match}
        if omitted:
            out["omitted"] = sorted(set(omitted))
        return out
