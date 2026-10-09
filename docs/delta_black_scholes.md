# Le delta de Black–Scholes : dérivation complète

Référence pour l'étape 1 (E0) : on a vendu un call européen de strike $K$ et de maturité $T$, et on le couvre avec le sous-jacent.

Notations : $\tau = T - t$ est le temps restant, $\varphi(x) = \frac{1}{\sqrt{2\pi}}e^{-x^2/2}$ la densité de la loi normale centrée réduite, et $N(x) = \int_{-\infty}^x \varphi(u)\thinspace du$ sa fonction de répartition. Dans les formules, $\mathrm{PnL}$ désigne le P&L.

---

## 1. Le modèle : $S_t$ est log-normal sous $\mathbb{P}$ et sous $\mathbb{Q}$

L'hypothèse de Black–Scholes porte sur la probabilité historique $\mathbb{P}$ :

$$
dS_t = \mu S_t\thinspace dt + \sigma S_t\thinspace dW_t ,
$$

avec un actif sans risque $dB_t = rB_t\thinspace dt$.

**Résolution.** On applique Itô à $f(S) = \ln S$, avec $f'(S) = 1/S$ et $f''(S) = -1/S^2$ :

$$
d\ln S_t = \frac{dS_t}{S_t} - \frac{1}{2}\frac{(dS_t)^2}{S_t^2}
= \mu\thinspace dt + \sigma\thinspace dW_t - \frac{1}{2}\sigma^2\thinspace dt ,
$$

car $(dS_t)^2 = \sigma^2 S_t^2\thinspace dt$. En intégrant :

$$
S_t = S_0 \exp\Big(\big(\mu - \tfrac{\sigma^2}{2}\big)t + \sigma W_t\Big).
$$

**Sous $\mathbb{Q}$.** Par Girsanov, $W_t^{\mathbb{Q}} = W_t + \frac{\mu - r}{\sigma}t$ est un mouvement brownien sous $\mathbb{Q}$, et $dS_t = rS_t\thinspace dt + \sigma S_t\thinspace dW_t^{\mathbb{Q}}$. Donc $S_t$ est log-normal **sous les deux probabilités**, avec la même volatilité $\sigma$ : seule la tendance change ($\mu$ sous $\mathbb{P}$, $r$ sous $\mathbb{Q}$).
- $\mathbb{P}$ décrit les trajectoires réelles : c'est sous $\mathbb{P}$ qu'on mesure le P&L de la couverture.
- $\mathbb{Q}$ est un outil de calcul pour le prix (section 5).
- Le delta ne dépend pas de $\mu$ (section 4). Dans nos simulations, on peut donc prendre $\mu = r = 0$, et $\mathbb{P} = \mathbb{Q}$.

---

## 2. Itô sur le prix de l'option

On cherche le prix du call sous la forme $C(t, S_t)$, avec $C$ de classe $C^{1,2}$. Par Itô :

$$
dC = \partial_t C\thinspace dt + \partial_S C\thinspace dS_t + \frac{1}{2}\partial_{SS} C\thinspace (dS_t)^2
= \Big(\partial_t C + \mu S\thinspace \partial_S C + \frac{1}{2}\sigma^2 S^2\thinspace \partial_{SS} C\Big)dt + \sigma S\thinspace \partial_S C\thinspace dW_t .
$$

---

## 3. Le portefeuille du vendeur : réplication

**Ce que détient réellement le vendeur.** À $t = 0$, il encaisse la prime $C_0$. Ensuite, à chaque instant, il détient $\delta_t$ actions et place le reste en cash au taux $r$. On note $X_t$ la valeur de ce portefeuille, avec $X_0 = C_0$ :

$$
X_t = \underbrace{\delta_t S_t}_{\text{actions}} + \underbrace{(X_t - \delta_t S_t)}_{\text{cash}} .
$$

Le portefeuille est **autofinancé** : on n'injecte ni ne retire d'argent, donc ses variations viennent seulement des actions et des intérêts :

$$
dX_t = \delta_t\thinspace dS_t + r\thinspace (X_t - \delta_t S_t)\thinspace dt .
$$

De façon équivalente, avec les valeurs actualisées $\tilde X_t = e^{-rt}X_t$ et $\tilde S_t = e^{-rt}S_t$ :

$$
d\tilde X_t = \delta_t\thinspace d\tilde S_t
\qquad\Longrightarrow\qquad
X_t = e^{rt}\Big(C_0 + \int_0^t \delta_u\thinspace d\tilde S_u\Big).
$$

Le cash n'est donc **pas** simplement $C_0 e^{rt}$ : il varie à chaque achat ou vente d'actions. La richesse totale vaut $C_0 e^{rt}$ plus les gains de trading actualisés.

**P&L du vendeur à l'échéance :**

$$
\mathrm{PnL} = X_T - (S_T - K)^+ .
$$

Avec $r = 0$ et un rebalancement quotidien, c'est exactement la formule de E0 : $\mathrm{PnL} = C_0 + \sum_t \delta_t\thinspace (S_{t+1} - S_t) - (S_T - K)^+$.

**Objectif : répliquer.** On cherche $\delta$ tel que $X_t = C(t, S_t)$ pour tout $t$, et donc $\mathrm{PnL} = 0$. Il faut que $dX_t$ et $dC$ coïncident. On réécrit $dX_t$ avec $dS_t$ :

$$
dX_t = \big(rX_t + (\mu - r)\thinspace \delta_t S_t\big)dt + \sigma S_t\thinspace \delta_t\thinspace dW_t .
$$

**Termes en $dW_t$.** On compare avec la section 2 :

$$
\boxed{\ \delta_t = \partial_S C(t, S_t)\ }
$$

**Termes en $dt$** (avec $X_t = C$ et $\delta_t = \partial_S C$) :

$$
rC + (\mu - r)\thinspace S\thinspace \partial_S C = \partial_t C + \mu S\thinspace \partial_S C + \frac{1}{2}\sigma^2 S^2\thinspace \partial_{SS} C .
$$

Les termes $\mu S\thinspace \partial_S C$ se simplifient des deux côtés.

---

## 4. L'EDP de Black–Scholes

Il reste :

$$
\boxed{\ \partial_t C + rS\thinspace \partial_S C + \frac{1}{2}\sigma^2 S^2\thinspace \partial_{SS} C - rC = 0, \qquad C(T, S) = (S - K)^+ \ }
$$

**Remarque importante** : $\mu$ a disparu. Le prix et la couverture ne dépendent pas du rendement espéré de l'action, seulement de $\sigma$.

**Lien avec le raisonnement « $\Pi = -C + \delta S$ » des manuels.** La quantité $\Pi_t = X_t - C(t, S_t)$ est la position nette du vendeur : son portefeuille moins l'option qu'il doit. La réplication dit que $\Pi_t = 0$ pour tout $t$. Le raccourci des manuels, qui écrit $d\Pi = -dC + \delta\thinspace dS$ en traitant $\delta$ comme constant, est justifié précisément par la condition d'autofinancement écrite sur $X_t$ ci-dessus.

---

## 5. Résolution : la formule de Black–Scholes

Par Feynman–Kac, la solution de l'EDP s'écrit comme une espérance sous la probabilité risque-neutre $\mathbb{Q}$, sous laquelle $dS_t = rS_t\thinspace dt + \sigma S_t\thinspace dW_t^{\mathbb{Q}}$ :

$$
C(t, S) = e^{-r\tau}\thinspace \mathbb{E}^{\mathbb{Q}}\big[(S_T - K)^+ \thinspace \big|\thinspace  S_t = S\big].
$$

D'après la section 1 (avec $\mu$ remplacé par $r$) :

$$
S_T = S\thinspace \exp\Big(\big(r - \tfrac{\sigma^2}{2}\big)\tau + \sigma\sqrt{\tau}\thinspace Z\Big), \qquad Z \sim \mathcal{N}(0, 1).
$$

**Quand l'option est-elle exercée ?** $S_T \gt  K$ équivaut à

$$
Z \gt  \frac{\ln(K/S) - (r - \sigma^2/2)\tau}{\sigma\sqrt\tau} = -d_2,
\qquad\text{avec}\quad d_2 = \frac{\ln(S/K) + (r - \sigma^2/2)\tau}{\sigma\sqrt\tau}.
$$

On découpe $(S_T - K)^+ = S_T\thinspace \mathbf 1_{Z \gt  -d_2} - K\thinspace \mathbf 1_{Z \gt  -d_2}$.

**Second terme :** $\mathbb{E}\big[K\thinspace \mathbf 1_{Z \gt  -d_2}\big] = K\thinspace \mathbb{P}(Z \gt  -d_2) = K\thinspace N(d_2)$.

**Premier terme :**

$$
\mathbb{E}\big[S_T\thinspace \mathbf{1}_{Z \gt  -d_2}\big]
= S\thinspace e^{(r - \sigma^2/2)\tau} \int_{-d_2}^{+\infty} e^{\sigma\sqrt\tau\thinspace z}\thinspace \varphi(z)\thinspace dz .
$$

On complète le carré : $\sigma\sqrt\tau\thinspace z - \frac{z^2}{2} = -\frac{(z - \sigma\sqrt\tau)^2}{2} + \frac{\sigma^2\tau}{2}$, donc $e^{\sigma\sqrt\tau z}\varphi(z) = e^{\sigma^2\tau/2}\thinspace \varphi(z - \sigma\sqrt\tau)$. Ainsi

$$
\mathbb{E}\big[S_T\thinspace \mathbf{1}_{Z \gt  -d_2}\big]
= S\thinspace e^{r\tau} \int_{-d_2}^{+\infty} \varphi(z - \sigma\sqrt\tau)\thinspace dz
= S\thinspace e^{r\tau}\thinspace N(d_2 + \sigma\sqrt\tau) = S\thinspace e^{r\tau}\thinspace N(d_1),
$$

avec

$$
d_1 = d_2 + \sigma\sqrt\tau = \frac{\ln(S/K) + (r + \sigma^2/2)\tau}{\sigma\sqrt\tau}.
$$

**Résultat :**

$$
\boxed{\ C(t, S) = S\thinspace N(d_1) - K e^{-r\tau}\thinspace N(d_2)\ }
$$

---

## 6. Le delta : $\partial_S C = N(d_1)$

On dérive en $S$, sans oublier que $d_1$ et $d_2$ dépendent de $S$ :

$$
\partial_S C = N(d_1) + S\thinspace \varphi(d_1)\thinspace \partial_S d_1 - K e^{-r\tau}\thinspace \varphi(d_2)\thinspace \partial_S d_2 .
$$

Comme $d_2 = d_1 - \sigma\sqrt\tau$, on a $\partial_S d_1 = \partial_S d_2 = \frac{1}{S\sigma\sqrt\tau}$. Il suffit donc de montrer que

$$
S\thinspace \varphi(d_1) = K e^{-r\tau}\thinspace \varphi(d_2).
$$

**Preuve de l'identité.** $d_2^2 = (d_1 - \sigma\sqrt\tau)^2 = d_1^2 - 2d_1\sigma\sqrt\tau + \sigma^2\tau$, donc

$$
\varphi(d_2) = \varphi(d_1)\thinspace \exp\Big(d_1\sigma\sqrt\tau - \frac{\sigma^2\tau}{2}\Big).
$$

Or $d_1\sigma\sqrt\tau = \ln(S/K) + (r + \sigma^2/2)\tau$, donc l'exposant vaut $\ln(S/K) + r\tau$, et

$$
\varphi(d_2) = \varphi(d_1)\thinspace \frac{S}{K}\thinspace e^{r\tau}
\quad\Longleftrightarrow\quad
K e^{-r\tau}\thinspace \varphi(d_2) = S\thinspace \varphi(d_1). \qquad\blacksquare
$$

Les deux derniers termes se compensent, et il reste :

$$
\boxed{\ \delta_t = \partial_S C(t, S_t) = N(d_1) \ }
$$

**Interprétation.** $\delta_t \in (0, 1)$. Il est proche de 1 quand l'option est très dans la monnaie ($S \gg K$), proche de 0 quand elle est très hors de la monnaie, et vaut environ $1/2$ à la monnaie. Quand $\tau \to 0$, il tend vers $\mathbf 1_{S \gt  K}$ : la courbe se raidit à l'approche de l'échéance.

---

## 7. Ce qui change en couverture quotidienne (notre expérience)

En pratique, on rebalance tous les $\Delta t = 1$ jour et non en continu. Prenons $r = 0$ pour simplifier. Le gamma vaut

$$
\Gamma = \partial_{SS} C = \frac{\varphi(d_1)}{S\sigma\sqrt\tau},
$$

et l'EDP donne $\partial_t C = -\frac{1}{2}\sigma^2 S^2\thinspace \Gamma$. Sur un pas, avec $\delta = N(d_1)$ et un développement de Taylor :

$$
\Delta\Pi \approx -\partial_t C\thinspace \Delta t - \tfrac{1}{2}\Gamma\thinspace (\Delta S)^2
= \frac{1}{2}\Gamma S^2\Big(\sigma^2\Delta t - \big(\tfrac{\Delta S}{S}\big)^2\Big).
$$

- L'**espérance** est nulle, car $\mathbb{E}\big[(\Delta S/S)^2\big] \approx \sigma^2\Delta t$.
- La **variance** n'est pas nulle : avec $(\Delta S/S)^2 \approx \sigma^2\Delta t\thinspace Z^2$ et $\mathrm{Var}(Z^2) = 2$, chaque pas apporte une variance $\frac{1}{2}\Gamma^2 S^4\sigma^4\Delta t^2$. En sommant les $T/\Delta t$ pas, l'écart-type total est d'ordre $\sqrt{\Delta t}$.

Un ordre de grandeur classique (Derman & Kamal, 1999) est $\mathrm{std}(\mathrm{PnL}) \approx \sqrt{\pi/4}\ \sigma\thinspace \mathcal{V}/\sqrt{n}$, avec $\mathcal{V} = S\sqrt{T}\varphi(d_1)$ le vega et $n$ le nombre de rebalancements. Pour $S = K = 100$, $\sigma$ = 20 %, $T = 30$ jours et $n = 30$, cela donne environ $0{,}37$, pour une prime d'environ $2{,}3$.

**Conséquences pour E0 :**
1. Même la « vraie » couverture $N(d_1)$ laisse un P&L aléatoire. Le repère chiffré est l'écart-type du P&L obtenu avec $\delta = N(d_1)$ sur les trajectoires de test.
2. En temps discret, la couverture qui minimise exactement $\mathbb{E}[\mathrm{PnL}^2]$ n'est pas tout à fait $N(d_1)$, mais l'écart est petit (d'ordre $\Delta t$). On attend donc que le réseau colle à $N(d_1)$ à quelques centièmes près, et que son écart-type de P&L soit au niveau de celui de $N(d_1)$, voire très légèrement en dessous.

---

## 8. Lien avec le réseau

Le MLP $f_\theta$ remplace la formule : $\delta_t = f_\theta\big(\tau_t/T,\ \ln(S_t/K)\big)$. On minimise

$$
L(\theta) = \frac{1}{M}\sum_{m=1}^{M}\Big(C_0 - (S^{(m)}_T - K)^+ + \sum_{t} f_\theta\big(\tau_t/T, \ln(S^{(m)}_t/K)\big)\thinspace \big(S^{(m)}_{t+1} - S^{(m)}_t\big)\Big)^2
$$

sur $M$ trajectoires simulées. Le réseau ne voit ni $\sigma$, ni la formule de la section 6. S'il retrouve la courbe $N(d_1)$, c'est que la simulation, le P&L, le gradient (autodiff PyTorch) et l'optimiseur sont corrects.
