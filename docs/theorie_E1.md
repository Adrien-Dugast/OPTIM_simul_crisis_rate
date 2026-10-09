# E1 : théorie financière et calculs

But de E1 : construire le monde simulé des taux (OAT et Bund) et calculer **à la main** les couvertures de référence. Elles serviront de plancher et de plafond pour juger le réseau dans E2 et E3.

Notations : $t$ en années, pas quotidien $\Delta t = 1/252$, horizon $T = 3$ mois (63 jours). « pb » = point de base = $10^{-4}$. Toutes les formules sont écrites pour un nominal de 100.

---

## 1. Prix d'une obligation zéro-coupon et taux actuariel

Une obligation zéro-coupon de maturité $M$ paie 100 en $M$. Si son taux actuariel (en composition continue) vaut $y$, son prix est

$$
P = 100\thinspace  e^{-y M}.
$$

Le taux et le prix varient en sens inverse : quand le taux monte, le prix baisse.

On utilise deux obligations de maturité 10 ans :
- **OAT** (État français), au taux $y^{FR}_t = r_t + s_t$ ;
- **Bund** (État allemand), au taux $y^{DE}_t = r_t$.

Ici $r_t$ est le taux allemand 10 ans et $s_t = y^{FR}_t - y^{DE}_t$ le **spread France**. Le spread mesure la prime de risque que les investisseurs demandent pour prêter à la France plutôt qu'à l'Allemagne. Il vaut environ 50 à 80 pb sur la période récente, et c'est lui qui bouge lors des crises politiques françaises.

Comme on tient les obligations pendant 3 mois, leur maturité restante diminue : à la date $t$, elle vaut $M - t$. Les prix sont donc

$$
P^{F}_t = 100\thinspace  e^{-(r_t + s_t)(M - t)},
\qquad
P^{B}_t = 100\thinspace  e^{-r_t (M - t)}.
$$

**Simplifications à mentionner dans le rapport.**
- Zéro-coupons au lieu d'obligations à coupons : seule la duration change un peu, la logique est la même.
- Le future Bund est remplacé par l'obligation allemande elle-même. Le vrai future a un panier de titres livrables, un facteur de conversion et une base, qu'on ignore.
- On ignore le coût de financement (portage) sur 3 mois.

---

## 2. Sensibilité au taux : duration et DV01

On dérive le prix par rapport au taux :

$$
\frac{\partial P}{\partial y} = -(M - t)\thinspace  P .
$$

- La **duration** (modifiée) est $D = -\frac{1}{P}\frac{\partial P}{\partial y}$. Pour un zéro-coupon, $D = M - t$, soit environ 10 ans.
- Le **DV01** est la variation de prix pour une hausse de 1 pb : $\mathrm{DV01} = D \cdot P \cdot 10^{-4}$.

Pour une petite variation $\Delta y$, un développement de Taylor à l'ordre 2 donne

$$
\Delta P \approx -D P\thinspace  \Delta y + \frac{1}{2} D^2 P\thinspace  (\Delta y)^2 .
$$

Le second terme est la **convexité**. Sur un jour, $\Delta y$ est de quelques pb, donc ce terme est négligeable devant le premier. On le garde dans la simulation (on calcule les prix exacts), mais on l'ignore dans les calculs de couverture.

**Ordres de grandeur** (avec $r = 2{,}5$ % et $s = 70$ pb) : $P^F \approx 72{,}6$, $P^B \approx 77{,}9$, et le DV01 de l'OAT vaut environ $0{,}073$ pour 100 de nominal.

---

## 3. Le modèle de taux à deux régimes

### 3.1 Taux allemand : processus de Vasicek

$$
dr_t = a\thinspace (b - r_t)\thinspace dt + \sigma_r\thinspace  dW^1_t .
$$

- $b$ est la moyenne de long terme, $a$ la vitesse de retour vers $b$, et $\sigma_r$ la volatilité.
- On modélise directement le taux 10 ans par ce processus d'Ornstein–Uhlenbeck. C'est une simplification : dans le vrai modèle de Vasicek, c'est le taux court qui suit cette dynamique, et le taux 10 ans s'en déduit. Pour notre problème de couverture, seule la dynamique du taux 10 ans compte.

**Discrétisation exacte.** La solution d'un processus d'Ornstein–Uhlenbeck est gaussienne. Sur un pas $\Delta t$ :

$$
r_{t+\Delta t} = b + (r_t - b)\thinspace  e^{-a \Delta t} + \sigma_r \sqrt{\frac{1 - e^{-2a\Delta t}}{2a}}\  Z^1_t ,
\qquad Z^1_t \sim \mathcal{N}(0, 1).
$$

C'est exact quel que soit le pas, donc préférable au schéma d'Euler.

### 3.2 Spread France : régime calme

$$
ds_t = \kappa\thinspace (\bar s - s_t)\thinspace dt + \sigma_s\thinspace  dW^2_t ,
$$

discrétisé de la même façon. En calme, le spread est peu volatil et presque indépendant du taux allemand.

### 3.3 Régime de crise

Dans une proportion $p$ des trajectoires, une crise politique démarre à une date $\tau_c$ tirée uniformément dans $[0, T]$, et dure jusqu'à la fin de l'horizon. À partir de $\tau_c$ :
1. **Saut** du spread : $s_{\tau_c} \leftarrow s_{\tau_c} + J$, avec $J$ uniforme entre 20 et 40 pb. C'est l'ordre de grandeur de juin 2024, quand le spread 10 ans est passé d'environ 50 à 80 pb en quelques jours après la dissolution.
2. **Volatilité accrue** : $\sigma_s$ passe de la valeur calme à la valeur de crise.
3. **Nouveau niveau moyen** : $\bar s$ monte, le spread ne revient pas tout de suite à son niveau d'avant.
4. **Fuite vers la qualité** : les investisseurs vendent l'OAT et achètent le Bund. Le spread monte pendant que le taux allemand baisse, donc la corrélation entre $\Delta r$ et $\Delta s$ devient négative.

**Comment simuler la corrélation.** On tire deux gaussiennes indépendantes $Z^1, Z^{\perp}$, et on pose

$$
Z^2 = \rho\thinspace  Z^1 + \sqrt{1 - \rho^2}\  Z^{\perp},
$$

avec $\rho = \rho_{\text{calme}}$ ou $\rho = \rho_{\text{crise}}$ selon le régime. On a alors $\mathrm{Corr}(Z^1, Z^2) = \rho$.

### 3.4 Paramètres proposés

Ce sont des ordres de grandeur raisonnables, pas une calibration.

| Paramètre | Calme | Crise |
|---|---|---|
| $b$ (moyenne du taux allemand) | 2,5 % | idem |
| $a$ (retour à la moyenne) | 0,5 par an | idem |
| $\sigma_r$ (volatilité du taux allemand) | 80 pb par an, soit 5 pb par jour | idem |
| $\bar s$ (niveau moyen du spread) | 65 pb | 85 pb |
| $\kappa$ (retour du spread) | 2 par an | 1 par an |
| $\sigma_s$ (volatilité du spread) | 15 pb par an, soit 0,9 pb par jour | 60 pb par an, soit 3,8 pb par jour |
| $\rho$ (corrélation $\Delta r$, $\Delta s$) | 0 | −0,5 |
| saut $J$ à l'entrée en crise | — | 20 à 40 pb |

---

## 4. La position et son P&L

On détient **une OAT** (nominal 100) et on **vend** $h_t$ Bunds chaque jour, avec $h_t \geq 0$ en pratique. Sans coûts de transaction ni financement, le P&L sur l'horizon vaut

$$
\mathrm{PnL} = \underbrace{P^F_T - P^F_0}_{\text{OAT}} \  - \  \sum_{t} h_t \big(P^B_{t+\Delta t} - P^B_t\big).
$$

Sur un pas, avec la duration (section 2) et en notant $A_t = D_t P^F_t$ et $B_t = D_t P^B_t$ (pour 100 de nominal, $A \approx 10 \times 72{,}6 = 726$) :

$$
\Delta \mathrm{PnL}_t \approx -A_t\thinspace  (\Delta r_t + \Delta s_t) + h_t\thinspace  B_t\thinspace  \Delta r_t .
$$

Le premier terme est la perte de l'OAT quand les taux français montent. Le second est le gain de la vente de Bund quand les taux allemands montent.

---

## 5. Les trois couvertures de référence

### 5.1 Aucune couverture ($h = 0$)

$$
\mathrm{Var}(\Delta \mathrm{PnL}) = A^2\thinspace  \mathrm{Var}(\Delta r + \Delta s) = A^2 \big(\sigma_r^2 + \sigma_s^2 + 2\rho\thinspace  \sigma_r \sigma_s\big)\Delta t .
$$

On subit tout le risque de taux, allemand et spread.

### 5.2 Couverture en duration (DV01)

On choisit $h$ pour annuler l'exposition au taux allemand $\Delta r$ :

$$
h^{\mathrm{DV01}}_t = \frac{A_t}{B_t} = \frac{P^F_t}{P^B_t} \approx 0{,}93 .
$$

(Les deux durations sont égales ici, car les deux obligations ont la même maturité.) Il reste

$$
\Delta \mathrm{PnL}_t \approx -A_t\thinspace  \Delta s_t ,
$$

c'est-à-dire **le risque de spread pur**. C'est la couverture standard des desks, et elle est optimale en temps calme.

### 5.3 Couverture de variance minimale (« oracle »)

On cherche le $h$ qui minimise la variance du P&L sur un pas, en connaissant le régime :

$$
V(h) = \mathrm{Var}\big(-A(\Delta r + \Delta s) + h B\thinspace  \Delta r\big)
= (hB - A)^2\thinspace  \mathrm{Var}(\Delta r) - 2A(hB - A)\thinspace  \mathrm{Cov}(\Delta r, \Delta s) + A^2\thinspace  \mathrm{Var}(\Delta s).
$$

On annule la dérivée en $h$ :

$$
V'(h) = 2B(hB - A)\thinspace  \mathrm{Var}(\Delta r) - 2AB\thinspace  \mathrm{Cov}(\Delta r, \Delta s) = 0
\quad\Longrightarrow\quad
h^{\ast} = \frac{A}{B}\Big(1 + \frac{\mathrm{Cov}(\Delta r, \Delta s)}{\mathrm{Var}(\Delta r)}\Big) = h^{\mathrm{DV01}}\thinspace (1 + \beta),
$$

avec $\beta = \frac{\mathrm{Cov}(\Delta r, \Delta s)}{\mathrm{Var}(\Delta r)} = \rho\thinspace  \frac{\sigma_s}{\sigma_r}$, le coefficient de régression de $\Delta s$ sur $\Delta r$. La variance résiduelle vaut alors

$$
V(h^{\ast}) = A^2\thinspace  \sigma_s^2\thinspace  (1 - \rho^2)\thinspace  \Delta t .
$$

**Interprétation.**
- **En calme**, $\rho \approx 0$, donc $\beta \approx 0$ et $h^{\ast} = h^{\mathrm{DV01}}$ : la couverture en duration est optimale.
- **En crise**, $\rho \lt  0$. Le jour où le spread s'écarte, l'OAT perd, mais le Bund monte (son taux baisse), et comme on est vendeur de Bund, on perd aussi sur la couverture. Il faut donc **vendre moins de Bund** : $h^{\ast} \lt  h^{\mathrm{DV01}}$. Avec les paramètres du tableau, $\beta = -0{,}5 \times 3{,}8 / 5 \approx -0{,}375$, donc $h^{\ast} \approx 0{,}63\thinspace  h^{\mathrm{DV01}}$.
- La partie $\sqrt{1 - \rho^2}$ du risque de spread reste **impossible à couvrir** avec le Bund seul : c'est le risque de base.

**Remarque.** Minimiser la variance pas à pas revient à minimiser la variance totale, parce que les accroissements quotidiens sont presque indépendants (le retour à la moyenne est lent devant un jour). C'est donc bien la couverture optimale au sens de la MSE quand le régime est connu.

### 5.4 Le saut n'est pas couvrable

Le jour de l'entrée en crise, le spread saute de $J$. Aucune position en Bund ne compense ce saut, puisqu'il ne touche pas le taux allemand. La perte vaut environ

$$
A\thinspace  J \approx 726 \times 0{,}003 \approx 2{,}2 \quad \text{pour } J = 30 \text{ pb},
$$

soit environ 3 % de la valeur de l'OAT. C'est du même ordre que tout le risque résiduel accumulé sur 3 mois en crise (voir le tableau ci-dessous). Conséquence pour la perte CVaR : la queue de distribution est dominée par ce saut, et le réseau ne peut pas l'éliminer. Il peut seulement mieux couvrir **après** le saut.

### 5.5 Ordres de grandeur

Écart-type du P&L **quotidien** pour 100 de nominal, avec les paramètres du tableau :

| Couverture | Calme | Crise |
|---|---|---|
| Aucune | 0,37 | 0,33 |
| Duration ($h^{\mathrm{DV01}}$) | 0,069 | 0,27 |
| Oracle ($h^{\ast}$) | 0,069 | 0,24 |

- En calme, la couverture en duration divise le risque par plus de 5.
- En crise, elle ne fait presque plus rien (0,27 contre 0,33 sans couverture), et l'oracle fait gagner environ 13 % d'écart-type. C'est cet écart que le réseau doit combler en apprenant à reconnaître la crise.

---

## 6. Ce que voit le réseau

- **Entrées**, toutes normalisées (centrées-réduites sur le jeu d'entraînement) :
  - temps restant $\tau_t / T$ ;
  - niveau du taux allemand $r_t$ et niveau du spread $s_t$ ;
  - volatilité réalisée du spread sur 10 jours, en log : c'est le signal de crise ;
  - position précédente $h_{t-1}$ (utile seulement avec des coûts de transaction).
- **Sortie** : le ratio $h_t / h^{\mathrm{DV01}}_t$ plutôt que $h_t$ lui-même. La sortie est alors d'ordre 1, et la cible vaut 1 en calme et environ 0,63 en crise.
- **Perte** : MSE ou CVaR du P&L total (section 4).
- Le réseau ne reçoit ni le régime, ni $\rho$, ni $\sigma_s$ : il doit les déduire des trajectoires.

---

## 7. Vérifications de E1, avant d'entraîner les optimiseurs

1. **Trajectoires** : tracer quelques trajectoires calmes et de crise de $r_t$ et $s_t$, et vérifier les ordres de grandeur (volatilité quotidienne, taille du saut, corrélation empirique dans chaque régime).
2. **Couvertures de référence** : calculer par Monte-Carlo l'écart-type et la CVaR du P&L total pour les trois couvertures, séparément sur les trajectoires calmes et de crise. Les chiffres doivent être cohérents avec la section 5.5.
3. **Premier entraînement** : un réseau entraîné avec Adam sur des trajectoires toutes calmes doit retrouver $h_t / h^{\mathrm{DV01}}_t \approx 1$.

Ensuite seulement, on passe à E2 et E3 (comparaison des optimiseurs, balayage de $p$).
