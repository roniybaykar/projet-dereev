# Projet DEREEV — Détection automatique des réévaluations d’antibiotiques

## 1. Présentation

DEREEV est un projet de traitement automatique du langage médical réalisé en collaboration avec l’Hôpital Foch.

L’objectif est de détecter automatiquement, dans des notes médicales rédigées en français, la présence ou l’absence d’une réévaluation d’un traitement antibiotique.

Le problème est formulé comme une classification binaire :

* **classe 0** : absence de réévaluation ;
* **classe 1** : présence d’une réévaluation.

Le projet contient actuellement :

* une préparation des données à partir d’un fichier Excel ;
* une baseline **TF-IDF + régression logistique** ;
* un pipeline de fine-tuning de **CamemBERT-bio** ;
* une validation croisée stratifiée ;
* cinq répétitions avec des seeds d’entraînement différentes ;
* une optimisation du seuil de décision sur les prédictions out-of-fold ;
* une évaluation finale sur un jeu de test fixe ;
* le calcul du F1 de la classe positive, de la PR-AUC, de la précision, du recall, de l’accuracy et des matrices de confusion.

---

## 2. Modèle Transformer actuel

Le modèle utilisé par défaut est :

```text
Nom affiché       : CamemBERT-bio
Identifiant HF    : almanach/camembert-bio-base
Dossier de sortie : camembert_bio
```

Ces valeurs sont définies dans `src/config.py` :

```python
MODEL_DISPLAY_NAME = "CamemBERT-bio"
MODEL_NAME = "almanach/camembert-bio-base"
MODEL_OUTPUT_NAME = "camembert_bio"
```

Pour tester un autre modèle dans le même pipeline, il faut modifier ces trois valeurs ensemble afin d’éviter d’écraser les résultats d’un modèle précédent.

Exemple avec DoctoBERT-fr :

```python
MODEL_DISPLAY_NAME = "DoctoBERT-fr"
MODEL_NAME = "doctolib-lab/doctobert-fr-base"
MODEL_OUTPUT_NAME = "doctobert_fr"
```

La compatibilité exacte d’un modèle avec `AutoModelForSequenceClassification` et sa longueur maximale de contexte doivent être vérifiées avant son lancement.

---

## 3. Organisation générale du pipeline

```text
Fichier Excel brut
        |
        v
src/prepare_data.py
        |
        +--> data/processed/train_cv.csv
        |
        +--> data/processed/test.csv
                    |
                    +-------------------------------+
                    |                               |
                    v                               v
          main.py — Transformer          src/tfidf_logistic.py
                    |                               |
                    +---------------+---------------+
                                    |
                                    v
                  folds fixes partagés entre les modèles
                                    |
                                    v
                 résultats OOF, seuils et modèles sauvegardés
                                    |
                                    v
                         src/evaluate.py
                                    |
                                    v
                    évaluation finale sur test.csv
```

---

## 4. Protocole expérimental : protocole A strict

Le benchmark suit un protocole conçu pour comparer les modèles de manière équitable.

### 4.1. Séparation externe train/test

Le fichier brut est séparé une seule fois en :

* `train_cv.csv` : données utilisées pour l’entraînement et la validation croisée ;
* `test.csv` : jeu de test final, conservé à part jusqu’à l’évaluation finale.

La séparation est stratifiée et utilise :

```python
TEST_SIZE = 0.15
RANDOM_SEED = 42
```

Le jeu de test représente donc 15 % des données.

Il ne doit pas être utilisé pour :

* choisir les hyperparamètres ;
* sélectionner un modèle ;
* choisir le seuil de décision ;
* effectuer l’early stopping.

### 4.2. Une seule répartition fixe des folds

À l’intérieur de `train_cv.csv`, une unique répartition stratifiée en `K` folds est créée avec :

```python
CV_SPLIT_SEED = 42
```

Pour le benchmark principal :

```python
K_FOLDS = 5
```

Cette affectation est sauvegardée dans un manifeste partagé :

```text
data/processed/cv_splits/fixed_split_seed_42_folds_5.csv
```

Chaque ligne de `train_cv.csv` reçoit définitivement un numéro de fold.

### 4.3. Cinq seeds utilisées uniquement pour l’entraînement

Les seeds d’entraînement sont :

```python
TRAINING_SEEDS = [42, 123, 456, 789, 2026]
```

Elles contrôlent notamment :

* l’initialisation de la tête de classification ;
* l’ordre des données et des batchs ;
* le dropout ;
* les sources d’aléatoire de NumPy ;
* les sources d’aléatoire de PyTorch ;
* les sources d’aléatoire de Transformers.

Elles **ne modifient jamais la composition des folds**.

Le protocole est donc :

```text
une répartition fixe en 5 folds
× 5 seeds d’entraînement
= 25 entraînements par modèle
```

Exemple :

```text
Seed d’entraînement 42
├── fold fixe 1
├── fold fixe 2
├── fold fixe 3
├── fold fixe 4
└── fold fixe 5

Seed d’entraînement 123
├── exactement le même fold 1
├── exactement le même fold 2
├── exactement le même fold 3
├── exactement le même fold 4
└── exactement le même fold 5
```

### 4.4. Garantie des mêmes splits entre tous les modèles

Le module `src/cv_protocol.py` est responsable de la création et du rechargement du manifeste.

Le manifeste est créé une seule fois avant la boucle sur les seeds. Tous les entraînements suivants rechargent la même affectation :

```text
row_index → numéro de fold
```

Ainsi, pour une valeur donnée de `--folds` :

* les cinq seeds utilisent exactement les mêmes indices de train et de validation ;
* TF-IDF utilise les mêmes folds que CamemBERT-bio ;
* les futurs modèles doivent recharger le même manifeste ;
* aucun `StratifiedKFold` n’est recréé dans la boucle des seeds.

Un fingerprint est calculé à partir de `train_cv.csv`.

Si le contenu ou l’ordre du dataset change, le programme refuse de réutiliser silencieusement un ancien manifeste incompatible.

> Pour un benchmark comparable, tous les modèles doivent être lancés avec le même fichier `train_cv.csv`, le même `CV_SPLIT_SEED` et le même nombre de folds.

---

## 5. Validation out-of-fold

Pour chaque seed d’entraînement, les cinq modèles produisent des prédictions sur leurs folds de validation respectifs.

Chaque observation de `train_cv.csv` est ainsi prédite par un modèle qui ne l’a pas utilisée pendant son entraînement.

Ces prédictions sont appelées **prédictions out-of-fold**, ou **OOF**.

Les prédictions OOF des cinq folds sont rassemblées pour obtenir, pour chaque seed :

* un F1 positif OOF ;
* une PR-AUC OOF ;
* une précision OOF ;
* un recall OOF ;
* une accuracy OOF ;
* une matrice de confusion OOF.

---

## 6. Métriques

### F1 de la classe positive

Le F1 de la classe positive mesure le compromis entre :

* la précision sur la classe `1` ;
* le recall sur la classe `1`.

Il s’agit de la métrique principale lorsque l’objectif est de détecter correctement les réévaluations.

### PR-AUC

La PR-AUC résume la courbe précision-recall.

Elle est particulièrement utile lorsqu’il existe un déséquilibre entre les classes.

Contrairement au F1, la PR-AUC ne dépend pas d’un seuil fixe.

### Autres métriques

Le pipeline calcule également :

* la précision ;
* le recall ;
* l’accuracy ;
* la matrice de confusion ;
* le nombre de prédictions positives ;
* le nombre réel de cas positifs ;
* la log loss lors de l’évaluation finale.

---

## 7. Optimisation du seuil

Le seuil standard est :

```text
0.50
```

Le programme teste également les seuils compris entre :

```text
0.05 et 0.95
```

avec un pas de :

```text
0.01
```

Le seuil qui maximise le F1 positif est choisi uniquement à partir des prédictions OOF de la seed concernée.

Le jeu de test n’est jamais utilisé pour choisir ce seuil.

Pour chaque seed, le programme conserve donc :

* les métriques avec le seuil standard `0.50` ;
* le meilleur seuil trouvé sur les OOF ;
* les métriques OOF avec le seuil optimisé.

En cas d’égalité entre plusieurs seuils, le seuil le plus proche de `0.50` est privilégié.

---

## 8. Évaluation finale sur le jeu de test

Pour chaque seed d’entraînement :

1. les cinq modèles correspondant aux cinq folds sont chargés ;
2. chaque modèle calcule une probabilité de classe `1` sur `test.csv` ;
3. les probabilités des cinq modèles sont moyennées ;
4. les métriques sont calculées au seuil `0.50` ;
5. le seuil optimisé sur les OOF de cette seed est appliqué.

On obtient ainsi un résultat test indépendant pour chacune des cinq seeds.

Le résultat final est présenté sous la forme :

```text
moyenne ± écart-type entre les cinq seeds
```

Exemple :

```text
F1 positif test : 0.8120 ± 0.0090
PR-AUC test      : 0.8640 ± 0.0060
```

L’écart-type utilisé est l’écart-type d’échantillon avec :

```python
ddof = 1
```

Le script calcule également un ensemble secondaire regroupant les probabilités des différentes seeds.

Le résultat principal du benchmark reste cependant la moyenne et l’écart-type des cinq répétitions indépendantes.

---

## 9. Paramètres du Transformer

Les paramètres actuels sont définis dans `src/config.py` :

```python
MAX_LENGTH = 512
BATCH_SIZE = 16
EPOCHS = 12
LEARNING_RATE = 2e-5

EARLY_STOPPING_PATIENCE = 2
EARLY_STOPPING_THRESHOLD = 0.001
```

### Signification

* `MAX_LENGTH` : longueur maximale des séquences tokenisées ;
* `BATCH_SIZE` : nombre d’exemples par lot ;
* `EPOCHS` : nombre maximal d’époques ;
* `LEARNING_RATE` : taux d’apprentissage ;
* `EARLY_STOPPING_PATIENCE` : nombre d’époques sans amélioration avant arrêt ;
* `EARLY_STOPPING_THRESHOLD` : amélioration minimale nécessaire pour être considérée comme significative.

Le meilleur checkpoint est sélectionné selon le F1 de validation au seuil `0.50`.

### Pondération des classes

Le pipeline utilise une `CrossEntropyLoss` pondérée.

Les poids de classes sont recalculés à partir du jeu d’entraînement de chaque fold afin de réduire l’effet du déséquilibre entre les classes `0` et `1`.

---

## 10. Installation

### Prérequis

* Python ;
* Git si le projet est cloné depuis un dépôt ;
* un GPU CUDA fortement recommandé pour les modèles Transformers ;
* suffisamment d’espace disque pour sauvegarder 25 modèles par architecture.

### Windows PowerShell

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### Linux ou serveur

```bash
python3 -m venv venv
source venv/bin/activate
python3 -m pip install --upgrade pip
pip install -r requirements.txt
```

Toutes les commandes suivantes doivent être exécutées depuis la racine du projet.

---

## 11. Préparation des données

Le fichier Excel brut attendu est défini dans `src/config.py` :

```text
data/raw/dereev_data_4_deduplicated.xlsx
```

Les colonnes attendues sont :

```text
cn_notes
reev_classe
```

Lancer la préparation :

```bash
python3 -m src.prepare_data
```

Le script :

1. charge le fichier Excel ;
2. vérifie la présence des colonnes ;
3. retire les valeurs manquantes ;
4. applique le nettoyage des textes ;
5. vérifie que les labels valent uniquement `0` ou `1` ;
6. réalise une séparation stratifiée train/test ;
7. sauvegarde les fichiers suivants :

```text
data/processed/train_cv.csv
data/processed/test.csv
```

Si ces fichiers existent déjà et que le dataset ne doit pas changer, il n’est pas nécessaire de relancer la préparation.

---

## 12. Vérification syntaxique

Avant un long entraînement, vérifier les fichiers principaux :

```bash
python3 -m py_compile main.py src/config.py src/cv_protocol.py src/evaluate.py src/tfidf_logistic.py
```

Si aucune erreur ne s’affiche, la vérification syntaxique est réussie.

---

## 13. Entraînement de CamemBERT-bio

### Test rapide avec une seule seed

Cette commande lance cinq entraînements, un par fold fixe :

```bash
python3 -m main --mode cv --folds 5 --seeds 42 --overwrite
```

Elle permet de vérifier le bon fonctionnement du pipeline avant de lancer les 25 fine-tunings.

### Entraînement complet recommandé

```bash
python3 -m main --mode cv --folds 5 --overwrite
```

Cette commande utilise automatiquement :

```text
42, 123, 456, 789, 2026
```

et lance :

```text
5 seeds × 5 folds fixes = 25 fine-tunings
```

### Liste personnalisée de seeds

```bash
python3 -m main --mode cv --folds 5 --seeds 42,123,456 --overwrite
```

### Autre nombre de folds

```bash
python3 -m main --mode cv --folds 3 --overwrite
```

Cette commande lance :

```text
5 seeds × 3 folds fixes = 15 entraînements
```

Pour le benchmark officiel, il est recommandé de conserver :

```text
--folds 5
```

Un nombre différent de folds crée un manifeste distinct.

Tous les modèles comparés doivent être lancés avec la même valeur de `--folds`.

### Option `--overwrite`

Sans `--overwrite`, le programme s’arrête si le dossier de résultats du modèle existe déjà.

Avec `--overwrite`, les anciens résultats du modèle courant sont supprimés avant le nouvel entraînement.

Les manifests de folds partagés ne sont pas supprimés.

---

## 14. Évaluation finale de CamemBERT-bio

Après l’entraînement complet :

```bash
python3 -m src.evaluate --folds 5
```

Avec les paramètres par défaut, cette commande évalue les seeds :

```text
42, 123, 456, 789 et 2026
```

La commande courte suivante est équivalente lorsque `K_FOLDS = 5` :

```bash
python3 -m src.evaluate
```

### Évaluer un test réalisé avec une seule seed

Après :

```bash
python3 -m main --mode cv --folds 5 --seeds 42 --overwrite
```

utiliser :

```bash
python3 -m src.evaluate --folds 5 --seeds 42
```

### Cohérence des paramètres

Le nombre de folds et la liste de seeds fournis à `evaluate` doivent correspondre aux entraînements présents.

Après un entraînement avec trois folds :

```bash
python3 -m main --mode cv --folds 3 --overwrite
```

l’évaluation doit être :

```bash
python3 -m src.evaluate --folds 3
```

Le script d’évaluation n’entraîne aucun modèle.

Il charge les modèles sauvegardés et calcule leurs prédictions sur `test.csv`.

---

## 15. Baseline TF-IDF + régression logistique

La baseline utilise exactement le même manifeste de folds fixe que le Transformer.

### Lancement complet

```bash
python3 -m src.tfidf_logistic --folds 5 --overwrite
```

### Test avec une seule seed

```bash
python3 -m src.tfidf_logistic --folds 5 --seeds 42 --overwrite
```

### Paramètres disponibles

```text
--folds N
--seeds 42,123,456,789,2026
--max-features N
--min-df N
--ngram-max 1|2|3
--c VALEUR
--max-iter N
--overwrite
```

Valeurs par défaut :

```text
max_features = 50 000
min_df       = 2
ngram_range  = unigrammes + bigrammes
C            = 1.0
max_iter     = 2 000
class_weight = balanced
solver       = liblinear
```

Exemple personnalisé :

```bash
python3 -m src.tfidf_logistic \
  --folds 5 \
  --max-features 75000 \
  --min-df 2 \
  --ngram-max 2 \
  --c 1.0 \
  --max-iter 3000 \
  --overwrite
```

La baseline est beaucoup plus rapide que le fine-tuning Transformer.

Selon le solveur et les données, les différentes seeds peuvent produire des résultats identiques ou presque identiques.

Cela indique simplement une faible sensibilité de cette baseline à l’aléatoire.

---

## 16. Affichage dans le terminal

Le programme affiche notamment :

* le nom du modèle utilisé ;
* l’identifiant Hugging Face ;
* le chemin du dossier de sortie ;
* la seed du split fixe ;
* la liste des seeds d’entraînement ;
* le nombre de folds ;
* le nombre total d’entraînements ;
* le chemin du manifeste partagé ;
* la progression globale, par exemple `8/25` ;
* les tailles des jeux train et validation ;
* la répartition des classes ;
* les poids de la loss ;
* le F1 positif ;
* la PR-AUC ;
* le recall ;
* la précision ;
* l’accuracy ;
* la matrice de confusion ;
* le meilleur seuil ;
* la durée de chaque fold ;
* la durée de chaque seed ;
* la moyenne, l’écart-type, le minimum et le maximum entre les seeds.

Le terminal confirme explicitement que les indices train/validation sont identiques pour toutes les seeds.

---

## 17. Organisation des résultats

### Manifeste de folds partagé

```text
data/processed/cv_splits/
├── fixed_split_seed_42_folds_5.csv
└── fixed_split_seed_42_folds_5.json
```

### Résultats du Transformer

```text
models/benchmark/camembert_bio/
├── seed_42/
│   ├── fold_1/
│   ├── fold_2/
│   ├── fold_3/
│   ├── fold_4/
│   ├── fold_5/
│   ├── validation_predictions.csv
│   ├── threshold_results.csv
│   ├── threshold_results.png
│   ├── precision_recall_curve.png
│   ├── decision_threshold.json
│   └── experiment_summary.json
├── seed_123/
├── seed_456/
├── seed_789/
├── seed_2026/
├── all_seeds_validation_results.csv
├── benchmark_summary.json
├── all_seeds_test_results.csv
├── all_seeds_ensemble_test_predictions.csv
├── all_seeds_ensemble_test_precision_recall_curve.png
└── test_benchmark_summary.json
```

### Résultats de TF-IDF

```text
models/benchmark/tfidf_lr/
├── seed_42/
│   ├── fold_1.joblib
│   ├── fold_2.joblib
│   ├── fold_3.joblib
│   ├── fold_4.joblib
│   ├── fold_5.joblib
│   ├── validation_predictions.csv
│   ├── test_predictions.csv
│   ├── threshold_results.csv
│   ├── threshold_results.png
│   ├── precision_recall_curve.png
│   └── experiment_summary.json
├── seed_123/
├── seed_456/
├── seed_789/
├── seed_2026/
├── all_seeds_results.csv
└── benchmark_summary.json
```

La dernière expérience Transformer est également enregistrée dans :

```text
models/latest_experiment.json
```

---

## 18. Fichiers principaux

### `main.py`

Entraîne le modèle Transformer avec plusieurs seeds sur les folds fixes partagés.

### `src/config.py`

Centralise :

* les chemins ;
* les seeds ;
* le nombre de folds ;
* le modèle ;
* les hyperparamètres.

### `src/cv_protocol.py`

Crée, sauvegarde, valide et recharge le manifeste fixe des folds.

### `src/evaluate.py`

Évalue sur le test les ensembles de modèles correspondant aux différentes seeds.

### `src/tfidf_logistic.py`

Entraîne et évalue la baseline TF-IDF + régression logistique avec les mêmes folds.

### `src/threshold_metrics.py`

Calcule les métriques, optimise le seuil et génère les courbes.

### `src/model.py`

Charge le modèle Transformer avec une tête de classification binaire.

### `src/prepare_data.py`

Prépare le dataset et crée `train_cv.csv` et `test.csv`.

### `src/data_loader.py`

Charge et tokenise les datasets pour le Transformer.

### `src/preprocessing.py`

Contient les fonctions de nettoyage et de gestion des abréviations.

---

## 19. Commandes principales

### Préparation des données

```bash
python3 -m src.prepare_data
```

### Test rapide du Transformer

```bash
python3 -m main --mode cv --folds 5 --seeds 42 --overwrite
python3 -m src.evaluate --folds 5 --seeds 42
```

### Benchmark complet du Transformer

```bash
python3 -m main --mode cv --folds 5 --overwrite
python3 -m src.evaluate --folds 5
```

### Baseline complète

```bash
python3 -m src.tfidf_logistic --folds 5 --overwrite
```

---

## 20. Benchmark multi-modèles

Pour comparer correctement :

```text
TF-IDF + LR
CamemBERT-bio
DoctoBERT-fr
DoctoModernBERT-fr
EuroBERT
BioMistral-LoRA
```

tous les modèles doivent utiliser :

* le même `train_cv.csv` ;
* le même `test.csv` ;
* le même manifeste de folds ;
* `CV_SPLIT_SEED = 42` ;
* `K_FOLDS = 5` ;
* les mêmes seeds d’entraînement ;
* les mêmes définitions de métriques ;
* une sélection du seuil uniquement sur les OOF ;
* aucune optimisation effectuée sur le jeu de test.

Le tableau final doit présenter au minimum :

```text
Modèle | F1 positif moyen ± écart-type | PR-AUC moyenne ± écart-type
```

Il est également possible d’ajouter :

* la précision ;
* le recall ;
* le F1 avec seuil optimisé ;
* le temps d’entraînement ;
* le nombre de paramètres ;
* la longueur maximale de contexte.

---

## 21. Reproductibilité

Le programme fixe les principales sources d’aléatoire de :

* Python ;
* NumPy ;
* PyTorch ;
* CUDA ;
* Transformers.

Les options suivantes sont également utilisées lorsque disponibles :

* algorithmes déterministes PyTorch ;
* désactivation du benchmark cuDNN ;
* désactivation de TF32 ;
* `full_determinism=True` ;
* `dataloader_num_workers=0`.

Malgré ces précautions, une reproductibilité bit à bit peut dépendre :

* de la version de PyTorch ;
* de la version de CUDA ;
* du GPU utilisé ;
* de la version de Transformers ;
* des versions des autres dépendances ;
* de certaines opérations GPU non entièrement déterministes.

Il est recommandé de conserver les versions exactes dans `requirements.txt` et de noter la configuration matérielle utilisée.

---

## 22. Ancien mode séquentiel

L’ancien mode :

```text
--mode sequential
```

et l’ancien paramètre :

```text
--calibration-size
```

ont été retirés de la version actuelle.

Le benchmark repose désormais uniquement sur des modèles indépendants entraînés en validation croisée avec :

* des folds fixes ;
* cinq seeds d’entraînement ;
* une évaluation OOF ;
* un jeu de test séparé.

La seule valeur acceptée pour `--mode` est :

```text
cv
```
