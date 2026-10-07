# Cas d'usage finance quantitative pour comparer GD et spectral GD

**Pourquoi la finance s'y prête bien.** Les matrices de rendements ont un spectre extrêmement étalé. Le **mode « marché »** domine : sa première valeur propre de la covariance vaut souvent 10 à 50 fois les suivantes. Viennent ensuite quelques facteurs sectoriels, puis un « bulk » de bruit décrit par Marchenko–Pastur.
C'est exactement le régime où GD apprend d'abord le marché et très lentement le reste, et où l'orthogonalisation pourrait aider.
L'autre point important : le **rapport signal/bruit est très faible**. Optimiser plus vite peut donc vouloir dire « surapprendre le bruit plus vite », ce qui fait une bonne discussion ERM.

**Données.** Pour la reproductibilité, le plus simple est de **simuler** un modèle à facteurs dont on connaît la vérité. En complément « réel », il y a les données gratuites de Kenneth French (portefeuilles et facteurs Fama–French). Il faudra vérifier depuis ton environnement qu'elles sont téléchargeables.
Légende de la difficulté : ★ = quelques heures, ★★ = un week-end, ★★★ = ambitieux.

---

## 1. Modèle à facteurs statistique (factorisation de la matrice des rendements) ★
$R \in \mathbb{R}^{T\times N} \approx F B^\top$, avec $F$ les facteurs ($T\times k$) et $B$ les expositions ($N\times k$). On minimise $\|R - FB^\top\|_F^2$ (en version profonde, c'est une PCA).
- **Pourquoi le spectral peut aider :** le facteur marché écrase les facteurs sectoriels. GD apprend en marches d'escalier (marché, puis secteurs, ...), alors que le spectral GD devrait les apprendre ensemble.
- **Mesures :** trajectoires des $\sigma_i(FB^\top)$, angle entre les sous-espaces appris et les vrais facteurs (en simulation).
- C'est la version finance de la piste B, et le plus simple pour démarrer.

## 2. Complétion de surface de volatilité implicite ou de courbe des taux ★★
Les cotations forment une matrice (strikes × maturités, ou dates × maturités pour les taux) dont certaines cases manquent : options illiquides, jours sans cotation.
Ces surfaces sont presque de faible rang ; pour la courbe des taux, Nelson–Siegel correspond à un rang d'environ 3 (niveau, pente, courbure).
- **Question ERM :** le biais implicite de GD vers le faible rang donne une surface lisse et plausible. Le spectral GD le conserve-t-il, ou produit-il des surfaces plus bruitées ?
- On mesure l'erreur sur les cotations cachées. C'est la version finance de la complétion de matrice.

## 3. Prédiction multi-actifs : VAR / régression à cible matricielle ★
$r_{t+1} = W r_t + \varepsilon_t$ (ou des caractéristiques $X_t$ à la place de $r_t$), avec $W \in \mathbb{R}^{N\times N}$.
- La covariance des entrées est très mal conditionnée à cause du mode marché, donc on contrôle κ « naturellement ».
- Le SNR est faible : on compare R² train et R² test en fonction du nombre d'itérations, pour savoir quelle méthode surapprend le plus vite. Ça se relie à l'arrêt précoce vu comme régularisation.

## 4. Estimation de matrice de précision / covariance ★★
On minimise la log-vraisemblance gaussienne $-\log\det(LL^\top) + \mathrm{tr}(S\,LL^\top)$ en paramétrant la précision par $L$ (Cholesky ou facteur libre). On peut ajouter une contrainte facteur + diagonale.
- Le spectre de $S$ est très étalé, et la Hessienne est mal conditionnée en direction des petites valeurs propres. Or c'est justement celles qui comptent pour les portefeuilles de variance minimale.
- **Évaluation financière :** variance hors échantillon du portefeuille de variance minimale construit avec l'estimateur obtenu. On peut comparer à la shrinkage de Ledoit–Wolf.

## 5. Markowitz : un contrôle négatif instructif ★
$\min_w\ w^\top\Sigma w - \lambda\mu^\top w$. C'est un vecteur, donc spectral = GD normalisé : aucun gain structurel attendu, malgré un κ(Σ) énorme.
- La version **matricielle** (K portefeuilles pour K aversions au risque, ou des portefeuilles imitant les facteurs) a une Hessienne de la forme $I\otimes\Sigma$.
- Ce contraste montre que le spectral égalise la géométrie *du gradient*, et non celle *de la Hessienne*.

## 6. IPCA : facteurs conditionnés par les caractéristiques ★★
Kelly, Pruitt & Su (2019) : $r_{i,t+1} = z_{i,t}^\top\Gamma f_{t+1} + \varepsilon$, avec $\Gamma \in \mathbb{R}^{L\times k}$ qui envoie les caractéristiques sur les expositions.
- C'est un problème bilinéaire $(\Gamma, f)$, classique en recherche quant, avec des paramètres matriciels naturels.
- Il se teste sur données simulées, ou sur les caractéristiques publiques d'Open Source Asset Pricing.

## 7. Deep hedging ★★
Buehler et al. (2019) : un MLP produit la position de couverture à chaque date. On minimise une mesure de risque (CVaR ou erreur quadratique) du P&L de couverture d'une option, sous Black–Scholes ou Heston simulés.
- C'est un vrai réseau, avec une perte non standard (de queue), et c'est 100 % simulé donc reproductible.
- On compare Muon, Adam et SGD : vitesse, stabilité et distribution du P&L. La CVaR rend les gradients portés par les **scénarios rares**, ce qui rejoint l'argument des « directions rares » amplifiées.

## 8. Calibration par surrogate neuronal ★★
Horvath, Muguruza & Tomas (2021), *Deep learning volatility* : un MLP apprend l'application (paramètres Heston ou rough Bergomi) → surface de prix ou de volatilité. Le surrogate sert ensuite à calibrer.
- C'est une régression lisse, très bien posée, avec beaucoup de données simulées. On peut y tester la vitesse et la précision atteignable à budget fixe.
- Cas où le SNR est élevé, à opposer aux cas 3 et 6 où il est faible.

## 9. Régimes rares (crises) ★★
On classe ou prédit avec un modèle linéaire ou un petit MLP, sur des données où les périodes de crise sont rares. Les directions de gradient associées aux crises ont alors de petites valeurs singulières.
- **Hypothèse :** le spectral apprend plus tôt le comportement en crise. On mesure l'erreur par régime. C'est l'analogue finance des classes déséquilibrées.

---

## Combinaisons qui font un bon récit

- **« Le facteur marché comme mauvais conditionnement »** : 1 (dynamique spectrale) + 3 (ERM et faible SNR) + 5 (contrôle). Tout se simule, et c'est le plus proche du cours.
- **« Faible rang et généralisation »** : 1 + 2. Visuel, avec une vraie question de biais implicite.
- **« Réseaux en finance »** : 7 (ou 8) + 9. Plus ML, avec l'argument des scénarios rares.

Ma recommandation : partir de **1 + 3 sur données simulées** (modèle à facteurs avec un facteur marché dominant), puis valider sur les portefeuilles de Kenneth French si le temps le permet.
