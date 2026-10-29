# Superconductor Screen

A small, open, repeatable screen for new superconductor candidates. Four times a day a GitHub
Actions job trains a model on known superconductors, generates new compositions, ranks them,
checks the best ones against the Materials Project, and publishes the results:

**Results:** https://jasonbrelsford.github.io/superconductor-screen/

## What it does

1. **Data.** The SuperCon-derived dataset of Hamidieh (2018), UCI Machine Learning Repository
   #464: about 21,000 measured superconductors, merged to about 15,000 unique compositions.
2. **Features.** Each formula becomes element fractions plus fraction-weighted statistics of
   element properties (atomic number, mass, electronegativity, radius, Mendeleev number, group,
   row, melting point, molar volume, conductivity, resistivity, valence electrons).
3. **Model.** XGBoost regression on Tc. Accuracy is reported two ways:
   * random 5-fold cross-validation (optimistic: near-duplicates leak between folds);
   * 5-fold cross-validation grouped by chemical system, so the model is always tested on
     element combinations it never saw. This is the number that matters.
   The five grouped-fold models form an ensemble; their spread is each prediction's uncertainty.
4. **Candidates.** For the best compound of every chemical system with Tc of 20 K or more, swap one
   element for a chemically similar one (same group and neighbouring row, neighbouring group, or
   another rare earth), fully or as 10% / 20% doping. Anions (O, N, F, S...) are only doped, never
   fully replaced. Only combinations of elements that appear in no training compound are kept.
   Each run also draws a fresh random batch of double substitutions (seeded by the UTC date and hour,
   so any run can be reproduced) and adds strong ones to `results/leaderboard.csv` with the date they
   were first seen.
5. **Ranking.** Three lists: highest score (prediction minus uncertainty), biggest predicted gain
   over the parent compound, and the best candidates outside the cuprates.
6. **Stability.** With a Materials Project key (`MP_API_KEY` repository secret), each listed
   composition is looked up: computed and near the convex hull, computed but unstable, or not yet
   computed (most doped formulas, whose stability is unknown).

## What it is not

These are predictions to test, not discoveries. A compound is a superconductor only when someone
makes it and measures zero resistance and the Meissner effect. Machine-learning Tc models are
known to over-predict for unfamiliar chemistry, the training data has inconsistent formulas
(for example oxygen counts), and the candidates inherit those quirks. Use the lists to choose
what to look at, and check the chemistry before anyone heads for a furnace.

## Run it yourself

```sh
python -m pip install -r requirements.txt
MP_API_KEY=your_key python run.py     # the key is optional
```

Output goes to `results/<date>/*.csv`, `results/latest.json` and `site/index.html`. Every run
is committed to `results/`, so the history of each prediction and its date are public.

## Credits

Data: K. Hamidieh, "A data-driven statistical model for predicting the critical temperature of a
superconductor", Computational Materials Science 154 (2018). Stability data: the Materials Project.
