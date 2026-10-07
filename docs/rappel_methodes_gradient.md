# Rappel : méthodes de gradient et « spectral gradient »

Cadre : on minimise $f(\theta)$. Pour les réseaux et la factorisation, les paramètres sont souvent des **matrices** $W \in \mathbb{R}^{m\times n}$, de gradient $G = \nabla_W f$.
Conditionnement (cas $L$-lisse, $\mu$-fortement convexe) : $\kappa = L/\mu$. Sur une quadratique $f(\theta)=\tfrac12\theta^\top H\theta - b^\top\theta$, $\kappa = \lambda_{\max}(H)/\lambda_{\min}(H)$.

**Fil conducteur :** presque toutes les méthodes ci-dessous sont une *descente la plus raide pour une certaine norme* :
$$d^\star = \arg\min_{d}\ \langle G, d\rangle + \tfrac{1}{2\eta}\|d\|^2 .$$
Changer la norme change la géométrie, donc la manière dont la méthode subit le conditionnement.

---

## 1. Descente de gradient (GD)
$$\theta_{k+1} = \theta_k - \eta \nabla f(\theta_k)$$
- Norme sous-jacente : $\ell_2$ (Frobenius pour une matrice).
- Avec $\eta = 1/L$ : $f(\theta_k)-f^\star = O\big((1-1/\kappa)^k\big)$, soit $O(\kappa\log(1/\varepsilon))$ itérations.
- Sur une quadratique, chaque direction propre $i$ se contracte au facteur $|1-\eta\lambda_i|$ : les petites valeurs propres sont lentes. **C'est le mécanisme du mauvais conditionnement.**

## 2. Momentum : Heavy-ball (Polyak) et Nesterov
- Heavy-ball : $\theta_{k+1} = \theta_k - \eta\nabla f(\theta_k) + \beta(\theta_k - \theta_{k-1})$.
- Nesterov : gradient évalué au point extrapolé $\theta_k + \beta(\theta_k-\theta_{k-1})$.
- Taux $O\big((1-1/\sqrt{\kappa})^k\big)$ : on passe de $\kappa$ à $\sqrt\kappa$. Heavy-ball est optimal sur les quadratiques mais sans garantie globale ; Nesterov est l'accélération prouvée en convexe.
- En apprentissage profond, on utilise la forme « buffer » $M_k = \beta M_{k-1} + G_k$, qui est aussi celle de Muon.

## 3. Gradient stochastique et minimisation du risque empirique (ERM)
$f(\theta) = \tfrac1N\sum_i \ell(\theta; x_i, y_i)$, SGD avec mini-lots : $\theta_{k+1} = \theta_k - \eta_k g_k$, avec $\mathbb{E}[g_k] = \nabla f$.
- Le bruit impose des pas décroissants (ou fixe un plancher d'erreur).
- **Biais implicite** : en régression linéaire sur-paramétrée, GD initialisé en 0 converge vers la solution de norme $\ell_2$ minimale ; en factorisation matricielle, il favorise les solutions de faible rang / faible norme nucléaire (Gunasekar et al., 2017). Changer l'optimiseur (donc la norme) change ce biais, ce qui est une question de généralisation et pas seulement de vitesse.

## 4. Préconditionnement et Newton
$$\theta_{k+1} = \theta_k - \eta P^{-1}\nabla f(\theta_k)$$
- C'est une descente la plus raide pour la norme $\|d\|_P = \sqrt{d^\top P d}$.
- Newton ($P = \nabla^2 f$) est invariant par changement de variables affine, donc insensible à $\kappa$ localement, mais coûte cher en grande dimension.
- Idéal : $P \approx H$, ce qui donne $\kappa(P^{-1/2}HP^{-1/2}) \approx 1$.

## 5. Méthodes adaptatives diagonales : AdaGrad, RMSProp, Adam
Adam (Kingma & Ba, 2015) :
$$m_k = \beta_1 m_{k-1} + (1-\beta_1)g_k,\quad v_k = \beta_2 v_{k-1} + (1-\beta_2)g_k^2,\quad \theta_{k+1} = \theta_k - \eta\,\frac{\hat m_k}{\sqrt{\hat v_k}+\epsilon}$$
- C'est un préconditionneur **diagonal**, coordonnée par coordonnée. AdamW découple la pénalité $\ell_2$.
- Avec $\beta_1=\beta_2=0$, on obtient **signSGD** : $\theta \leftarrow \theta - \eta\,\mathrm{sign}(g)$, qui est la descente la plus raide en norme $\ell_\infty$ (Bernstein & Newhouse, 2024).
- Limite : la diagonale ignore les corrélations entre coordonnées, donc la structure matricielle de $W$.

## 6. Descente la plus raide pour une norme quelconque
$$d^\star = \arg\min_{\|d\|\le 1}\langle G, d\rangle, \qquad \text{pas non normalisé : } \Delta = -\eta\,\|G\|_*\, d^\star$$
où $\|\cdot\|_*$ est la norme duale.

| Norme sur $d$ | Direction $d^\star$ | Norme duale | Méthode |
|---|---|---|---|
| $\ell_2$ / Frobenius | $-G/\|G\|_F$ | $\ell_2$ / Frobenius | GD (normalisé) |
| $\ell_\infty$ | $-\mathrm{sign}(G)$ | $\ell_1$ | signSGD, ≈ Adam |
| **spectrale** $\|\cdot\|_{2\to2}$ | $-UV^\top$ si $G = U\Sigma V^\top$ | nucléaire $\sum_i\sigma_i$ | **spectral GD, Muon** |

## 7. Spectral gradient / Muon
**Idée.** On prend la SVD réduite $G = U\Sigma V^\top$ et on remplace $\Sigma$ par l'identité :
$$W_{k+1} = W_k - \eta\, U V^\top \qquad (\text{ou } -\eta\,\|G\|_*\,UV^\top \text{ pour la version non normalisée}).$$
- $UV^\top$ est le **facteur polaire** de $G$, c'est-à-dire la matrice (semi-)orthogonale la plus proche de $G$ en norme de Frobenius.
- Toutes les directions singulières du gradient avancent à la même vitesse : les directions « rares » (petits $\sigma_i$) sont amplifiées et les directions dominantes freinées. C'est une sorte de blanchiment de la mise à jour.
- Pour un vecteur ($n=1$), $UV^\top = g/\|g\|$ : on retrouve le **GD normalisé**. L'effet spécifique n'apparaît donc qu'avec des paramètres matriciels.

**Muon** (Jordan et al., 2024) = momentum + orthogonalisation :
$$M_k = \beta M_{k-1} + G_k,\qquad O_k = \mathrm{NS}(M_k) \approx \mathrm{polar}(M_k),\qquad W_{k+1} = W_k - \eta\, s\, O_k$$
(avec Nesterov en option et un facteur d'échelle $s$ dépendant de la forme, par ex. $\sqrt{\max(1, m/n)}$).
- **Newton–Schulz** remplace la SVD par une itération polynomiale (GPU-friendly). On normalise d'abord $X_0 = M/\|M\|_F$, puis
  - version classique (cubique) : $X \leftarrow \tfrac32 X - \tfrac12 XX^\top X$, qui converge vers $UV^\top$ ;
  - version Muon (quintique, ~5 itérations) : $X \leftarrow aX + b(XX^\top)X + c(XX^\top)^2X$ avec $(a,b,c) = (3.4445, -4.7750, 2.0315)$. Plus rapide, mais ne donne qu'une orthogonalisation approchée (valeurs singulières dans ~[0.7, 1.2]).
- En pratique, Muon s'applique aux matrices des couches cachées ; les embeddings, la tête de sortie et les biais restent sous AdamW.
- Il est disponible dans les versions récentes de PyTorch (`torch.optim.Muon`, à vérifier selon ta version), ce qui va dans le sens de « use built-in optimizers ».

**Lien avec Shampoo.** Sans accumulation, $(GG^\top)^{-1/4}\, G\, (G^\top G)^{-1/4} = UV^\top$ : Muon est un Shampoo « sans mémoire ».

**Points d'attention pour le DM :**
- La norme du pas ne tend pas vers 0 quand $G \to 0$ (comme sign/GD normalisé). À pas fixe, on obtient des oscillations autour de l'optimum, ce qui demande un pas décroissant. C'est un point à discuter face aux résultats de convergence linéaire de GD.
- Sur $f(W) = \tfrac12\|XW - Y\|_F^2$, le mauvais conditionnement vient de $X^\top X$. L'orthogonalisation égalise les valeurs singulières de $G$, ce qui n'est **pas** la même chose que préconditionner par $(X^\top X)^{-1}$. Savoir quand l'un approxime l'autre est une bonne question d'étude.

## 8. Shampoo, SOAP, K-FAC (préconditionneurs matriciels)
- **Shampoo** (Gupta et al., 2018) : $L_k = L_{k-1} + G G^\top$, $R_k = R_{k-1} + G^\top G$, puis $W \leftarrow W - \eta\, L_k^{-1/4} G R_k^{-1/4}$. Préconditionneur de Kronecker, plus riche que la diagonale d'Adam.
- **SOAP** (Vyas et al., 2024) : Adam exécuté dans la base propre de Shampoo.
- **K-FAC** (Martens & Grosse, 2015) : approximation de Kronecker de la matrice de Fisher (gradient naturel).

## 9. ⚠️ Homonyme : la « spectral gradient method » de Barzilai–Borwein
En optimisation classique, « spectral gradient » désigne souvent **BB** (Barzilai & Borwein, 1988 ; Raydan, 1997 pour la version « spectral projected gradient ») :
$$\eta_k = \frac{s^\top s}{s^\top y},\quad s = \theta_k - \theta_{k-1},\ y = \nabla f(\theta_k) - \nabla f(\theta_{k-1}).$$
- $1/\eta_k$ est un quotient de Rayleigh de la Hessienne moyenne, d'où le nom « spectral ».
- Méthode non monotone, très efficace sur les quadratiques, avec une seule taille de pas scalaire (pas de géométrie matricielle).

Vu l'énoncé (matrices, réseaux, factorisation), le sujet vise très probablement Muon / la norme spectrale. Une confirmation rapide auprès de l'enseignant ne coûte rien, et comparer BB comme référence peut aussi être un ajout sympa.

## 10. Récapitulatif

| Méthode | Géométrie | Dépendance en $\kappa$ (quadratique) | Coût en plus de GD |
|---|---|---|---|
| GD | $\ell_2$ | $\kappa$ | — |
| Heavy-ball / Nesterov | $\ell_2$ + inertie | $\sqrt\kappa$ | négligeable |
| Adam | diagonale adaptative | aide si mal conditionné *selon les axes* | $O(d)$ |
| Spectral / Muon | norme spectrale | aide si le gradient est mal conditionné *spectralement* | SVD ou ~5 NS ($O(mn\min(m,n))$) |
| Shampoo | Kronecker | approche Newton pour les structures Kronecker | racines 4ᵉ matricielles |
| Newton | Hessienne | indépendant de $\kappa$ | inversion de Hessienne |
| BB | pas scalaire adaptatif | bon en pratique, non monotone | négligeable |

---

## Pistes d'expériences (pour un projet sympa)

**A. Régression linéaire matricielle à conditionnement contrôlé.**
$f(W) = \tfrac12\|XW - Y\|_F^2$ avec un spectre de $X^\top X$ choisi (κ = 10, 10², 10⁴) et une cible $W^\star$ de rang/spectre contrôlé. Comparer GD, GD + momentum, Adam, spectral GD (SVD exacte vs Newton–Schulz), voire BB. Mesurer le nombre d'itérations pour atteindre ε en fonction de κ, et l'effet du pas fixe vs décroissant. Contrôle de cohérence : avec $n=1$, spectral = GD normalisé.

**B. Factorisation matricielle : apprentissage incrémental des valeurs singulières.**
$\min_{A,B} \tfrac12\|AB^\top - M\|_F^2$ avec $M$ de faible rang et des valeurs singulières très étalées. GD apprend les modes un par un, en « marches d'escalier » (Saxe et al., 2014) ; les mises à jour spectrales devraient les apprendre de façon plus simultanée. On trace les trajectoires des $\sigma_i(AB^\top)$, ce qui donne une figure très parlante. Extension ERM : complétion de matrice (entrées partielles). Le biais implicite de GD vers le faible rang survit-il avec Muon ? Comparer l'erreur de test sur les entrées cachées.

**C. Petit MLP (MNIST ou sous-ensemble de CIFAR).**
Comparer Muon, AdamW et SGD + momentum avec un balayage du learning rate : robustesse au choix du pas, vitesse en époques et en temps réel. On peut aussi suivre le rang effectif ou le spectre des poids et des mises à jour au fil de l'entraînement, et faire une ablation sur le nombre d'itérations de Newton–Schulz.

A et B se combinent bien en un récit cohérent : conditionnement → dynamique spectrale → généralisation, le tout sur des modèles jouets reproductibles. C sert de validation « réseau de neurones ».

## Références principales
- Jordan et al. (2024), *Muon: An optimizer for hidden layers in neural networks* (blog).
- Bernstein & Newhouse (2024), *Old Optimizer, New Norm: An Anthology*.
- Liu et al. (2025), *Muon is Scalable for LLM Training*.
- Gupta, Koren & Singer (2018), *Shampoo*. Vyas et al. (2024), *SOAP*.
- Barzilai & Borwein (1988) ; Raydan (1997).
- Saxe, McClelland & Ganguli (2014) ; Gunasekar et al. (2017), *Implicit Regularization in Matrix Factorization*.
- Nesterov, *Lectures on Convex Optimization* (taux de GD et de l'accélération).
