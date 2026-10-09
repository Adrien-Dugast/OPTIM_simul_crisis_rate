# CLAUDE.md

Devoir maison d'optimisation du master M2MO (enseignant : Tam Le, 2026).
Question posée : *« When, and in what sense, do spectral gradient updates improve gradient-based optimization and machine learning? »*
Contraintes de l'énoncé : PyTorch obligatoire (différentiation automatique, optimiseurs intégrés quand ils existent), rapport final de 4 pages maximum (figures, références et paragraphe d'usage de l'IA compris), rapport intermédiaire d'une page, code en `.py` et/ou notebook reproductible.

## Sujet : couverture d'une position de taux en période de crise

Cas d'usage 7 (deep hedging) combiné avec une partie du cas 9 (régimes rares) de [`docs/cas_usage_finance_quant.md`](docs/cas_usage_finance_quant.md).

- **Position** : longue OAT 10 ans, couverte chaque jour par le future Bund (plus liquide).
- **Modèle de taux à deux facteurs** : taux allemand $r_t^{DE}$ (Vasicek, retour à la moyenne) + spread France $s_t$.
  - Régime calme : le spread bouge peu ; le ratio de couverture « duration OAT / duration Bund » suffit. C'est le **repère théorique**.
  - Régime de crise politique, de probabilité $p$ faible : le spread saute (sauts + volatilité accrue). La couverture laisse un risque de base que le réseau doit apprendre à gérer.
- **Couverture** : un petit MLP choisit la position $\delta_t = \mathrm{NN}(t, r_t^{DE}, s_t, \delta_{t-1})$.
  P&L final $= \Delta V^{OAT} + \sum_t \delta_t\thinspace \Delta F_t^{Bund} - \text{coûts de transaction}$.
- **Perte** : CVaR du P&L par la formule de Rockafellar–Uryasev,
  $\min_{\theta, w}\ w + \frac{1}{1-\alpha}\thinspace \mathbb{E}\big[(-\mathrm{PnL}_\theta - w)^+\big]$ ; la MSE sert de point de comparaison.
- **Intuition testée** : les trajectoires de crise sont rares dans chaque mini-lot, donc portées par de petites valeurs singulières du gradient des couches. GD/Adam apprend à couvrir le calme bien avant la crise ; l'orthogonalisation du spectral GD (Muon) devrait remonter ces directions faibles et apprendre plus tôt la couverture de crise.
- **Garde-fou** : le spectral GD ne prédit pas une crise. On étudie quelle méthode apprend une structure rare présente dans les données, et à quelle vitesse.

## Méthodes d'optimisation retenues

Rappel détaillé : [`docs/rappel_methodes_gradient.md`](docs/rappel_methodes_gradient.md).

- **SGD (+ momentum)** : référence norme $\ell_2$.
- **Adam / AdamW** : préconditionneur diagonal (norme $\approx \ell_\infty$).
- **Spectral GD / Muon** : descente la plus raide en norme spectrale. Pour $G = U\Sigma V^\top$, mise à jour selon $UV^\top$ ; Muon = momentum + orthogonalisation par Newton–Schulz. Utiliser `torch.optim.Muon` (PyTorch ≥ 2.9) pour les matrices de poids 2D, et Adam/AdamW pour les biais et paramètres scalaires (dont $w$ de la CVaR), comme le recommande Muon.
- Optionnel : spectral GD « exact » par SVD pour vérifier Newton–Schulz sur un petit cas.
- Homonyme à mentionner sans le traiter : Barzilai–Borwein (« spectral gradient » au sens d'un pas scalaire).

## Conventions de code

- **Langage** : Python 3.11+. **Librairies** : PyTorch (modèles, simulation vectorisée, optimiseurs), numpy, matplotlib. Pas d'autre dépendance sans raison.
- **Reproductibilité** : graine fixée partout (`torch.manual_seed`, `numpy.random.default_rng(seed)`), scénarios de test générés une fois et réutilisés pour toutes les méthodes, plusieurs graines par configuration (moyenne ± écart-type).
- **Organisation prévue** :
  - `src/simulation.py` : modèle de taux à régimes et prix OAT / future Bund ;
  - `src/hedging.py` : MLP de couverture, P&L, pertes CVaR et MSE ;
  - `src/optimizers.py` : construction des optimiseurs (SGD, Adam, Muon, SVD exact) ;
  - `src/experiments.py` : boucles d'entraînement et balayages, résultats en `results/*.csv` ;
  - `notebooks/rapport.ipynb` : figures du rapport à partir des résultats ;
  - `tests/` : tests rapides (ratio de duration retrouvé en régime calme, orthogonalité de la mise à jour Muon).
- **Entrées du réseau normalisées, d'ordre 1** (temps restant $\tau/T$, moneyness $\ln(S/K)$, taux centrés-réduits) ; ne pas donner au réseau les paramètres constants ni ceux du modèle. Une entrée mal mise à l'échelle crée un mauvais conditionnement qui fausse la comparaison des optimiseurs.
- **Perte toujours construite sur le P&L**, jamais sur l'écart à un delta de référence.
- Code et noms de variables en anglais, commentaires et docstrings en français ou en anglais (rester cohérent dans un même fichier). Fonctions courtes, sans état global.
- Les figures se génèrent par script ou notebook, jamais à la main.

## Rapport

- **Langue : français.**
- 4 pages maximum, figures comprises ; terminer par un paragraphe de 5 lignes maximum sur l'usage de l'IA, en indiquant explicitement les idées venues d'un LLM.
- Plan d'expériences et plan du rapport : [`docs/plan.md`](docs/plan.md) (E0 contrôle Black–Scholes, E0 bis crise rare sur l'action, puis taux OAT/Bund).
- Dérivation du delta de Black–Scholes : [`docs/delta_black_scholes.md`](docs/delta_black_scholes.md).
- Théorie de E1 (obligations, duration, modèle à deux régimes, couverture de variance minimale) : [`docs/theorie_E1.md`](docs/theorie_E1.md).

## Autres notes

- [`docs/taches_a_optimiser.md`](docs/taches_a_optimiser.md) : éventail de tâches génériques pour comparer GD et spectral GD.
