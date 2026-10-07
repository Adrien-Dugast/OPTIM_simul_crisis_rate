# Plan d'expériences et plan du rapport

Question : **dans un problème de couverture où la crise est rare, le spectral GD (Muon) apprend-il plus tôt, ou mieux, la couverture de crise que GD/Adam ?**
Cadre et notations : voir [`CLAUDE.md`](../CLAUDE.md).

---

## 1. Plan d'expériences

### E0. Validation du code (version action, contrôle)
- Couverture d'un call sous Black–Scholes, perte MSE, sans coûts.
- Attendu : le δ appris est proche du delta de Black–Scholes. Sert uniquement à vérifier la chaîne simulation → réseau → P&L → gradient.
- Test unitaire : la mise à jour de Muon est bien (approximativement) orthogonale, $\|UV^\top\|_2 \approx 1$, et coïncide avec la version SVD exacte sur une petite matrice.

### E1. Modèle de taux à deux facteurs et repère théorique
- Simulation : $r^{DE}$ Vasicek, spread $s$ à changement de régime (chaîne de Markov calme/crise, probabilité d'entrée en crise calibrée pour qu'une fraction $p$ des trajectoires contienne une crise), sauts du spread en crise. Ordres de grandeur grossièrement calés sur l'épisode de juin 2024 (spread OAT–Bund 10 ans passé d'environ 50 à 80 pb en quelques jours).
- Instruments : OAT 10 ans (zéro-coupon ou coupon fixe), future Bund approché par une obligation allemande ~10 ans.
- Repère : couverture au ratio de duration, calculé analytiquement. Vérifier qu'en régime calme le réseau le retrouve, et mesurer le risque de base résiduel en crise.

### E2. Expérience centrale : plan 2 × 2 (perte × optimiseur)
| | Adam (ou SGD+momentum) | Muon |
|---|---|---|
| **MSE** | | |
| **CVaR (α = 95 %)** | | |
- Même architecture (MLP 2–3 couches cachées de 32–64 neurones), mêmes scénarios, budget d'itérations fixe, taux d'apprentissage choisi par petite grille pour chaque méthode (même effort de réglage pour toutes).
- Question : CVaR (poids sur la queue côté **perte**) et Muon (côté **géométrie**) se substituent-ils ou se cumulent-ils ?

### E3. Balayage de la rareté p
- $p \in \{1\,\%, 2\,\%, 5\,\%, 10\,\%, 20\,\%\}$, perte CVaR, Adam vs Muon, 5 graines.
- Figure clé : le gain de Muon (écart de CVaR sur les trajectoires de crise à budget fixe, ou nombre d'itérations pour atteindre un seuil) en fonction de $p$.
- Attendu si l'intuition est juste : gain croissant quand la crise devient plus rare, puis nul quand $p$ est grand.

### E4. Diagnostic spectral (le « pourquoi »)
- Au fil de l'entraînement, valeurs singulières du gradient de la première couche, et part de la norme du gradient portée par les trajectoires de crise (gradient calculé séparément sur le sous-lot crise et le sous-lot calme, projeté sur les directions singulières).
- Vérifie (ou infirme) que la crise vit dans les petites valeurs singulières et que Muon l'amplifie.

### E5. ERM et généralisation (si le temps le permet)
- Peu de trajectoires de crise à l'entraînement : comparer CVaR de crise en train et en test au fil des itérations. Muon apprend-il plus vite la crise, ou surapprend-il plus vite les quelques crises vues ?
- Variante : momentum de Muon à 0 vs 0,95, pour relier au cours sur le momentum.

### Mesures communes
- CVaR, moyenne et écart-type du P&L, **séparément sur les trajectoires calmes et de crise**, sur un jeu de test fixe et grand.
- Courbes en fonction des itérations et du temps de calcul.
- Écart entre δ appris et ratio de duration en régime calme.
- Moyenne ± écart-type sur plusieurs graines.

### Ordre de travail
1. E0 + E1 (code validé) : base du rapport intermédiaire.
2. E2, puis E3.
3. E4 pour l'interprétation, E5 en option.

---

## 2. Plan du rapport (4 pages, en français)

1. **Introduction** (≈ 0,3 p) : question du DM, motivation (crise politique française et spread OAT–Bund), intuition « directions rares ».
2. **Méthodes** (≈ 0,6 p) : GD, Adam, spectral GD/Muon vus comme descentes les plus raides pour différentes normes ; Newton–Schulz ; lien avec le conditionnement.
3. **Problème de couverture** (≈ 0,6 p) : modèle de taux à deux facteurs à régimes, P&L, CVaR de Rockafellar–Uryasev, repère du ratio de duration.
4. **Expériences et résultats** (≈ 1,5 p) : validation (E0–E1, brièvement), plan 2 × 2 (tableau), balayage en $p$ (figure clé), diagnostic spectral (une figure).
5. **Discussion** (≈ 0,5 p) : quand et en quel sens le spectral aide ; limites (modèle simulé, réglage des pas, peu de crises réelles), lien avec l'ERM et la généralisation ; ouverture vers des données réelles (Banque de France, BCE).
6. **Références et usage de l'IA** (≈ 0,2 p) : Buehler et al. (2019), Rockafellar & Uryasev (2000), Jordan et al. (Muon, 2024), Bernstein & Newhouse (2024) ; paragraphe IA de 5 lignes maximum.

### Rapport intermédiaire (1 page)
Travail fait (E0–E1), difficultés, discussions avec d'autres étudiants, auto-évaluation et suite prévue (E2–E5).
