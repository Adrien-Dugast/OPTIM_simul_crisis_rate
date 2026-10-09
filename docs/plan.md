# Plan d'expériences et plan du rapport

Question : **dans un problème de couverture où la crise est rare, le spectral GD (Muon) apprend-il plus tôt, ou mieux, la couverture de crise que SGD ou Adam ?**
Cadre et notations : voir [`CLAUDE.md`](../CLAUDE.md). Dérivation du delta de Black–Scholes : [`delta_black_scholes.md`](delta_black_scholes.md).

**Fil conducteur** : le même code de deep hedging sert à toutes les expériences, et on ne change qu'une chose à chaque étape.

| Étape | Monde simulé | Crise rare ? | Réponse connue ? | Rôle dans le récit |
|---|---|---|---|---|
| E0 | Black–Scholes | non | oui, $N(d_1)$ | contrôle : pas de raison que Muon gagne |
| E0 bis | Black–Scholes à deux régimes | oui, proportion $p$ | oui, $N(d_1)$ avec la $\sigma$ du régime | la question du DM dans un cadre où l'on connaît la solution |
| E1–E3 | taux OAT/Bund à deux régimes | oui, proportion $p$ | en calme seulement | la même question dans le cadre réaliste et actuel |
| E4–E5 | idem | oui | — | pourquoi, et est-ce que ça généralise |

---

## 0. Ce qui est commun à toutes les expériences

- **Couverture** : un MLP $f_\theta$ (2 couches cachées de 64 neurones) donne la position $\delta_t$, décidée en $t$ avec l'information disponible en $t$ et tenue jusqu'à $t+1$. Le même réseau est appliqué à toutes les dates : on empile les dates dans un tenseur $(M, N, \text{entrées})$ et un seul passage donne tous les $\delta$.
- **Entrées normalisées, d'ordre 1** : temps restant $\tau_t/T = (T-t)/T$ et moneyness $\ln(S_t/K)$ pour l'action ; niveaux centrés-réduits pour les taux. Les paramètres constants ($K$, $T$, $r$) ne sont pas donnés au réseau, et les paramètres du modèle ($\sigma$, intensité des sauts) non plus : le réseau doit les découvrir dans les trajectoires.
  *Pourquoi* : une entrée brute comme $S_t \approx 100$ crée un mauvais conditionnement artificiel, qu'Adam corrige en partie (normalisation par coordonnée) alors que SGD et Muon le subissent. On comparerait alors les optimiseurs sur un défaut de mise à l'échelle, pas sur la crise. À mentionner dans le rapport.
- **Perte construite sur le P&L, jamais sur $\delta$** : $\mathrm{PnL} = p_0 + \sum_t \delta_t\thinspace \Delta X_t - \text{payoff}$, puis MSE ($\mathbb{E}[\mathrm{PnL}^2]$) ou CVaR (Rockafellar–Uryasev). La prime $p_0$ est un scalaire appris (avec Adam). On ne régresse pas sur la réponse connue, parce que dans le cas réaliste (taux en crise, CVaR) il n'en existe pas : E0 vérifie justement que cette perte retrouve la bonne réponse là où on la connaît.
- **Une itération** = un lot de $M$ trajectoires (par exemple 4 096), tous les $\delta$, une perte, un pas d'optimiseur. Jeu de test fixe et grand, commun à toutes les méthodes.
- **Optimiseurs comparés** : SGD + momentum, Adam, Muon (`torch.optim.Muon` sur les matrices de poids, Adam sur les biais, $p_0$ et $w$). Même budget d'itérations, taux d'apprentissage choisi par une petite grille pour chaque méthode, 5 graines.
- **Mesures, toujours séparées entre trajectoires calmes et trajectoires de crise** : écart à la couverture de référence quand elle existe, écart-type et CVaR du P&L, en fonction des itérations et du temps de calcul.

---

## 1. Plan d'expériences

### E0. Contrôle : Black–Scholes sans crise
- Call européen vendu, $S_0 = K = 100$, $\sigma$ = 20 %, $r = 0$, $T = 60$ jours, rebalancement quotidien, perte MSE, sans coûts.
- **Référence** : $\delta_t = N(d_1)$. Même cette couverture laisse une erreur de discrétisation : son écart-type de P&L sur le jeu de test est le niveau à atteindre.
- **Avec les trois optimiseurs.** Attendu : tous retrouvent $N(d_1)$ (courbes $\delta$ en fonction de $S$ à quelques dates), sans différence nette de vitesse. Si Muon gagnait déjà ici, son avantage dans les étapes suivantes ne pourrait pas être attribué à la crise.
- **Tests unitaires** : orthogonalisation de Newton–Schulz proche de $UV^\top$ obtenue par SVD sur une petite matrice ; P&L calculé à la main sur une trajectoire courte égal à celui du code.

### E0 bis. Crise rare sur l'action (Black–Scholes à deux régimes)
- Même call, même réseau, même perte. Dans une proportion $p$ des trajectoires, une crise démarre à une date aléatoire et dure jusqu'à l'échéance : saut de prix à l'entrée en crise (par exemple −8 %), puis volatilité de 50 % au lieu de 15 %.
- **Entrées** : on ajoute un signal de régime. D'abord l'indicateur de crise lui-même (le réseau doit apprendre une seconde fonction, vue rarement), puis, en variante plus difficile, la volatilité réalisée sur les 5 derniers jours.
- **Référence** : après l'entrée en crise, $\delta_t = N(d_1)$ calculé avec $\sigma$ = 50 % ; en calme, avec $\sigma$ = 15 %.
- **Mesures** : itération à laquelle l'écart $|\delta_\theta - \delta^{\mathrm{ref}}|$ passe sous un seuil, séparément sur les dates de crise et les dates calmes ; écart-type du P&L des trajectoires de crise.
- **Balayage** : $p$ ∈ {1 %, 2 %, 5 %, 10 %, 20 %}, perte MSE puis CVaR, SGD/Adam/Muon, 5 graines.
- **Figure clé possible du rapport** : gain de Muon (itérations gagnées pour apprendre le delta de crise) en fonction de $p$. Attendu si l'intuition est juste : gain croissant quand la crise devient plus rare, nul quand elle est fréquente.

### E1. Modèle de taux à deux régimes et couvertures de référence
Théorie et calculs détaillés : [`theorie_E1.md`](theorie_E1.md).

- **Simulation** (horizon 3 mois, pas quotidien) : taux allemand 10 ans $r_t$ (Vasicek, moyenne 2,5 %, volatilité ≈ 0,8 % par an) et spread France $s_t$.
  - Calme : $s_t$ oscille autour de 60–70 pb, volatilité ≈ 15 pb par an, corrélation faible avec $r_t$.
  - Crise (proportion $p$ des trajectoires, date aléatoire) : saut de +20 à +40 pb (ordre de grandeur de juin 2024), volatilité accrue, et **fuite vers la qualité** : corrélation négative entre $\Delta r$ et $\Delta s$.
- **Instruments** : OAT zéro-coupon 10 ans $P^{OAT} = e^{-10(r+s)}$, future Bund approché par $P^{B} = e^{-10r}$. Position : 1 OAT, on vend $\delta_t$ Bund.
- **Couvertures de référence** (calculées à la main) :
  - aucune couverture ;
  - couverture en duration : $\delta = -P^{OAT}/P^{B}$, optimale en calme ;
  - couverture « oracle » de variance minimale dans chaque régime : $\delta^{\ast} = -\frac{P^{OAT}}{P^{B}}\big(1 + \frac{\mathrm{cov}(\Delta r, \Delta s)}{\mathrm{var}(\Delta r)}\big)$. En crise, la covariance est négative, donc il faut vendre moins de Bund.
- **Point à dire dans le rapport** : le saut du spread lui-même n'est pas couvrable avec le Bund seul, il reste un risque de base. Ce que le réseau peut apprendre, c'est à reconnaître la crise (niveau du spread, dernières variations) et à ajuster son ratio.
- **Vérifications** : ordres de grandeur des trajectoires ; CVaR des trois couvertures de référence, en calme et en crise (plancher et plafond pour le réseau) ; un entraînement Adam qui retrouve le ratio de duration en calme.

### E2. Plan 2 × 2 sur les taux (perte × optimiseur)
| | Adam (ou SGD + momentum) | Muon |
|---|---|---|
| **MSE** | | |
| **CVaR (α = 95 %)** | | |
- Question : la CVaR (poids sur la queue du côté de la **perte**) et Muon (poids sur les directions rares du côté de la **géométrie**) se substituent-ils ou se cumulent-ils ?
- Mesure principale : part de l'écart entre la couverture en duration et l'oracle comblée sur les trajectoires de crise.

### E3. Rareté p sur les taux
- Même balayage de $p$ que E0 bis, avec la perte CVaR. On vérifie que la conclusion de E0 bis tient dans le cadre réaliste.

### E4. Diagnostic spectral (le « pourquoi »)
- Au fil de l'entraînement : valeurs singulières du gradient de la première couche, et part de ce gradient portée par les trajectoires de crise (gradient calculé séparément sur le sous-lot crise et le sous-lot calme, projeté sur les directions singulières).
- Vérifie, ou infirme, que la crise vit dans les petites valeurs singulières et que Muon l'amplifie. Faisable sur E0 bis comme sur les taux.

### E5. ERM et généralisation (si le temps le permet)
- Peu de trajectoires de crise à l'entraînement (jeu fixe au lieu de trajectoires fraîches à chaque itération) : CVaR de crise en train et en test au fil des itérations. Muon apprend-il la crise, ou surapprend-il les quelques crises vues ?
- Variante : momentum de Muon à 0 et à 0,95, pour relier au cours sur le momentum.

### Ordre de travail
1. E0 (code validé, contrôle) puis E0 bis : base du rapport intermédiaire, avec un premier balayage de $p$.
2. E1, puis E2 et E3.
3. E4 pour l'interprétation, E5 en option.

---

## 2. Plan du rapport (4 pages, en français)

1. **Introduction** (≈ 0,3 p) : question du DM, motivation (crise politique française et spread OAT–Bund), intuition des « directions rares ».
2. **Méthodes** (≈ 0,5 p) : SGD, Adam et spectral GD/Muon vus comme des descentes les plus raides pour différentes normes ; Newton–Schulz ; lien avec le conditionnement (dont la normalisation des entrées).
3. **Deep hedging** (≈ 0,5 p) : P&L, perte MSE et CVaR de Rockafellar–Uryasev, références $N(d_1)$ et ratio de duration.
4. **Résultats** (≈ 1,7 p) :
   - contrôle E0 (une phrase et une petite figure : tout le monde retrouve $N(d_1)$) ;
   - crise rare sur l'action, E0 bis (figure clé : gain en fonction de $p$) ;
   - taux OAT/Bund, E1–E3 (tableau 2 × 2, confirmation en $p$) ;
   - diagnostic spectral, E4 (une figure).
5. **Discussion** (≈ 0,5 p) : quand et en quel sens le spectral aide ; limites (modèles simulés, réglage des pas, peu de crises réelles) ; lien avec l'ERM et la généralisation ; ouverture vers des données réelles (Banque de France, BCE).
6. **Références et usage de l'IA** (≈ 0,2 p) : Buehler et al. (2019), Rockafellar & Uryasev (2000), Jordan et al. (Muon, 2024), Bernstein & Newhouse (2024) ; paragraphe IA de 5 lignes maximum.

### Rapport intermédiaire (1 page)
Travail fait (E0, E0 bis), difficultés, discussions avec d'autres étudiants, auto-évaluation et suite prévue (E1–E5).
