# training script

### Orga
Pour éviter un seul fichier monolithique, ça vaut le coup de séparer en modules dès le départ :

network (couches, forward, backward, activations)
metrics (accuracy, precision, recall, F1, confusion matrix)
history (sauvegarde/chargement des métriques par epoch en JSON)
train.py (orchestration)
compare_models.py (script séparé, bonus 1, réutilise history)

### Etapes
1. Charger et préparer les données
Lire data/data_train.csv et data/data_valid.csv (générés par split_data.py).
Séparer chaque fichier en features (les 30 colonnes numériques) et labels (diagnosis, M/B).
Convertir les labels en format numérique exploitable par le réseau : puisque le sujet impose une couche de sortie softmax, il faut encoder la cible en one-hot (ex: B → [1,0], M → [0,1]) plutôt qu'en un simple 0/1.
Normaliser les features. C'est une étape critique ici : radius_mean est de l'ordre de 15, area_mean de l'ordre de 600, smoothness_mean de l'ordre de 0.1. Sans normalisation (standardisation ou min-max), le gradient descent va très mal converger. Important : les paramètres de normalisation (moyenne/écart-type par colonne) doivent être calculés uniquement sur le train, puis appliqués tels quels au valid (sinon fuite d'information). Il faudra aussi les sauvegarder, car le programme de prédiction devra appliquer exactement la même transformation.

2. Définir l'architecture du réseau
Taille de la couche d'entrée = nombre de features (30).
Au moins 2 couches cachées (le sujet l'impose par défaut), avec un nombre de neurones configurable (via fichier ou arguments CLI, comme le montre l'exemple du sujet).
Couche de sortie = 2 neurones avec activation softmax (probabilité malin/bénin qui somme à 1).
Choisir une fonction d'activation pour les couches cachées (sigmoïde ou ReLU typiquement).

3. Initialiser les poids
Chaque couche a une matrice de poids W et un vecteur de biais b.
Les poids doivent être initialisés aléatoirement (petites valeurs, pas à zéro sinon toutes les unités apprennent la même chose) — les biais peuvent démarrer à zéro.

4. Implémenter le feedforward (propagation avant)
Pour chaque couche : Z = W·A_prev + b, puis A = activation(Z).
Garder en mémoire les Z et A de chaque couche, car ils seront nécessaires pendant la rétropropagation.
La dernière couche applique softmax au lieu de l'activation habituelle.

5. Implémenter la fonction de coût
Cross-entropy binaire (formule donnée dans le sujet) entre les probabilités prédites et les labels one-hot.

6. Implémenter la rétropropagation (backpropagation)
Calculer l'erreur en sortie (différence entre prédiction et label réel).
Remonter couche par couche via la règle de la chaîne pour calculer le gradient de l'erreur par rapport à chaque W et chaque b.
Réutiliser les Z/A mis en cache à l'étape feedforward.

7. Implémenter la descente de gradient
Une fois les gradients calculés, mettre à jour chaque W et b : W -= learning_rate * dW.

8. Boucle d'entraînement (epochs + batchs)
Découper le train en mini-batchs (--batch_size), les mélanger à chaque epoch.
Pour chaque batch : feedforward → coût → backprop → mise à jour des poids.
À la fin de chaque epoch : calculer loss et accuracy sur le train ET sur le valid (sans mise à jour des poids sur le valid, juste évaluation), et les afficher au format attendu par le sujet (epoch 01/70 - loss: ... - val_loss: ...).
Stocker l'historique de ces métriques (nécessaire pour les courbes finales).

9. À la fin de l'entraînement
Tracer les deux courbes d'apprentissage (loss et accuracy, train vs valid) — obligatoire selon le sujet.
Sauvegarder le modèle sur disque : l'architecture (nombre de couches, tailles, activations) + les poids/biais appris + les paramètres de normalisation, pour que le programme de prédiction puisse tout recharger tel quel.

### Bonus
À faire seulement une fois la partie obligatoire terminée et testée de bout en bout (le bonus n'est évalué que si l'obligatoire est parfait).
Tout repose sur le dict `history` (une liste par métrique, une entrée par epoch) déjà rempli par train.py.

1. Historique des métriques ✅ (core/history.py)
À la fin de train.py, sauvegarder `history` en JSON (ex: history/<nom>.json), avec les hyperparamètres (couches, activation, learning_rate, batch_size, epochs, seed) pour savoir à quel modèle correspond chaque fichier.
Ajouter un argument `--name` (ou nom généré automatiquement à partir des hyperparamètres).

2. Plusieurs métriques ✅ (core/metrics.py, reste à les réutiliser dans predict.py)
Module core/metrics.py codé à la main avec numpy : confusion matrix (TP/FP/FN/TN, M = classe positive), accuracy, precision, recall, F1.
Les calculer à chaque epoch sur train ET valid, les ajouter à `history`.
Les réutiliser dans predict.py pour afficher un rapport complet en plus de la binary cross-entropy.
Soutenance : expliquer pourquoi le recall compte ici (rater un cancer malin coûte plus cher qu'une fausse alerte).

3. Plusieurs courbes sur le même graphe
Script séparé src/tools/compare_models.py hist1.json hist2.json ... : charge les historiques et trace val_loss / val_accuracy de chaque modèle sur le même graphe, légende construite à partir des hyperparamètres.
Démo : comparer 24 24 vs 64 32 16, sigmoid vs relu, différents learning rates.

4. Early stopping (optionnel)
Arguments `--patience N` (désactivé par défaut, le sujet montre un entraînement jusqu'à la dernière epoch) et éventuellement `--min-delta`.
Dans la boucle : si val_loss s'améliore → garder une COPIE des poids (W.copy(), b.copy(), sinon on garde une référence qui continue d'être modifiée) et remettre le compteur à 0 ; sinon incrémenter le compteur, arrêter quand il atteint patience.
À la fin, restaurer les meilleurs poids avant la sauvegarde du modèle.