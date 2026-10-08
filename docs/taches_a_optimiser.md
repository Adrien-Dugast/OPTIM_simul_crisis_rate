# Quelles tâches optimiser pour comparer GD et spectral GD ?

Rappel : le spectral GD remplace $G = U\Sigma V^\top$ par $UV^\top$. Il ne peut donc différer de GD (normalisé) que si les paramètres sont des **matrices**, et il devrait aider surtout quand le gradient a des valeurs singulières **très étalées**.
Légende de la difficulté : ★ = quelques heures, ★★ = un week-end, ★★★ = plus ambitieux.

---

## 1. Contrôles et cas « théoriques »

**1.1 Fonctions test vectorielles (Rosenbrock, quadratique mal conditionnée en $\mathbb{R}^d$)** ★
Sur un vecteur, spectral = GD normalisé. On s'attend donc à ce qu'il **n'aide pas** au-delà de la normalisation.
C'est un contrôle négatif qui montre que l'effet vient de la structure matricielle.

**1.2 Quadratique à Hessienne de Kronecker : $f(W) = \tfrac12\Vert AWB - C\Vert _F^2$** ★
La Hessienne vaut $(BB^\top)\otimes(A^\top A)$ : on contrôle séparément le conditionnement « à gauche » et « à droite ».
Shampoo y est presque exact. Le spectral GD égalise les $\sigma_i$ du gradient mais ne préconditionne pas vraiment.
Question : quand $UV^\top$ approxime-t-il bien la direction de Newton ? On peut tracer le nombre d'itérations en fonction de $\kappa(A)$ et $\kappa(B)$. Ça se relie très directement au cours sur le conditionnement.

## 2. Problèmes convexes d'apprentissage (ERM)

**2.1 Régression linéaire multi-sorties à κ contrôlé** ★ (piste A)
Covariance des entrées choisie, cible $W^\star$ de spectre choisi. On étudie la vitesse en fonction de κ, le pas fixe vs décroissant et le momentum.

**2.2 Régression logistique multiclasse (softmax) sur MNIST ou Fashion-MNIST** ★
$W \in \mathbb{R}^{10\times 784}$, problème convexe sur données réelles, avec erreur train et test.
Avec un mini-lot de taille $b$, le gradient est de rang $\le b$ et très déséquilibré, donc l'orthogonalisation change vraiment la direction.
Bonne question ERM : la vitesse sur le risque empirique se traduit-elle en erreur de test ?

**2.3 Classes déséquilibrées / queue lourde** ★★
Même modèle, mais avec des fréquences de classes en loi de puissance.
Les classes rares contribuent peu aux grandes valeurs singulières du gradient, que le spectral GD « remonte ». On s'attend à ce qu'il apprenne les classes rares plus tôt, et on mesure la précision par classe au cours du temps. Des travaux récents (2025) étudient précisément cet effet pour Muon.

## 3. Problèmes matriciels non convexes

**3.1 Factorisation matricielle $\min\Vert AB^\top - M\Vert _F^2$** ★ (piste B)
GD apprend les valeurs singulières une par une, en marches d'escalier. Le spectral GD devrait les apprendre plus ensemble.
On obtient des figures très lisibles (trajectoires des $\sigma_i$).

**3.2 Complétion de matrice / matrix sensing** ★★
On observe seulement une partie des entrées. GD a un biais implicite vers le faible rang : ce biais survit-il avec le spectral ?
Ici, aller plus vite peut **nuire** à la généralisation. C'est un résultat intéressant, quel que soit le sens dans lequel il tombe.

**3.3 Réseau linéaire profond ($W_3W_2W_1$)** ★★
Le point selle en 0 et l'initialisation petite donnent des plateaux. Les mises à jour normalisées s'échappent-elles plus vite des plateaux ?
On peut comparer l'effet de la profondeur sur GD et sur le spectral GD.

**3.4 PCA par gradient : $\min_W \Vert X - WW^\top X\Vert _F^2$** ★
On apprend le sous-espace principal de dimension $k$. Le conditionnement est alors lié aux écarts entre valeurs propres (eigengap).
On regarde si le spectral GD est moins sensible à un petit eigengap.

## 4. Réseaux de neurones

**4.1 Mémoire associative linéaire** ★
Stocker des paires (clé → valeur), $W \approx \sum_i f_i e_i^\top$, avec des fréquences en loi de puissance et une perte cross-entropy.
C'est un jouet minimal pour les « faits rares » : on mesure le rappel en fonction de la fréquence. Bon pont entre 2.3 et les réseaux.

**4.2 Teacher–student MLP synthétique** ★★
Un réseau « professeur » aléatoire génère les labels, et un réseau élève l'apprend. La covariance des entrées est contrôlée.
On a un vrai réseau non linéaire, mais avec une vérité terrain connue, et on peut mesurer la généralisation exactement.

**4.3 Petit MLP ou CNN sur MNIST / CIFAR-10 (sous-ensemble)** ★★ (piste C)
On compare Muon, AdamW et SGD + momentum. Questions possibles : robustesse au learning rate, temps réel vs nombre d'itérations, rang effectif des poids.

**4.4 Mini-transformer sur caractères (tiny Shakespeare)** ★★★
C'est le cadre où Muon est réputé le plus fort. Il est plus coûteux, et l'énoncé ne valorise pas l'échelle : à garder en bonus.

---

## Combinaisons qui font un bon récit

- **Conditionnement :** 1.1 (contrôle) + 1.2 ou 2.1 (κ contrôlé) + 2.2 (données réelles). C'est la combinaison la plus proche du cours.
- **Dynamique spectrale et généralisation :** 3.1 + 3.2 (+ 3.3). C'est la plus visuelle.
- **Directions rares :** 2.3 + 4.1 (+ 4.3). C'est la plus originale et la plus « ML ».
