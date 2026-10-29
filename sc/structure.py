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
Doped (partial) substitutions are not computed: they need supercells and many configurations.

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
STEPS = 300
FMAX = 0.1


ANIONS = {"O", "N", "F", "S", "Se", "Te", "Cl", "Br", "I", "H", "B", "C", "P", "As"}
MINOR = 0.08        # cations below this share of the cation sites are treated as dopants
MAX_HOST_DISTANCE = 0.25


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
        """Closest stable-ish MP structure built from the parent's major elements, or (None, reason)."""
        major = self.major_elements(parent)
        docs = self.mpr.materials.summary.search(chemsys="-".join(sorted(major)),
                                                 fields=["material_id", "structure", "energy_above_hull", "nsites"])
        docs = [d for d in docs if d.nsites <= MAX_SITES and d.energy_above_hull is not None and d.energy_above_hull <= 0.1]
        if not docs:
            return None, "no host structure in Materials Project"
        best = min(docs, key=lambda d: (_frac_distance(d.structure.composition, parent), d.energy_above_hull))
        dist = _frac_distance(best.structure.composition, parent)
        if dist > MAX_HOST_DISTANCE:
            return None, f"closest host too different ({best.structure.composition.reduced_formula})"
        return best, ""

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

    def check(self, parent, swaps):
        if any(x < 1.0 for _, _, x in swaps):
            return {"status": "doped: not computed"}
        if any(a not in self.major_elements(parent) for a, _, _ in swaps):
            return {"status": "doped: not computed"}   # the swapped element is itself a minor dopant
        h, why = self.host(parent)
        if h is None:
            return {"status": why}
        s = h.structure.copy()
        present = {str(e) for e in s.composition.elements}
        mapping = {Element(a): Element(b) for a, b, _ in swaps if a in present}
        if len(mapping) != len(swaps):
            return {"status": "host structure lacks a swapped element", "host": str(h.material_id)}
        s.replace_species(mapping)
        relaxed, e = self._relax(s)
        els = [str(x) for x in relaxed.composition.elements]
        comps = self.competitor_energies(els)
        have = {str(x) for en in comps for x in en.composition.elements if en.composition.is_element}
        if set(els) - have:
            return {"status": "missing elemental reference", "host": str(h.material_id)}
        pd = PhaseDiagram(comps)
        cand = PDEntry(relaxed.composition, e)
        _, ehull = pd.get_decomp_and_e_above_hull(cand, allow_negative=True)
        ehull = round(float(ehull), 3)
        return {"status": "plausible" if ehull <= 0.1 else "likely unstable",
                "e_above_hull_eV": ehull, "computed_as": relaxed.composition.reduced_formula,
                "host": str(h.material_id)}
