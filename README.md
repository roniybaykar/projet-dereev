# DEREEV — Détection des réévaluations d’antibiotiques

## 1. Objectif du projet

DEREEV est un projet de traitement automatique du langage médical réalisé avec l’Hôpital Foch. Il analyse une note médicale en français et cherche à déterminer si elle contient une réévaluation d’un traitement antibiotique.

Le problème est une classification binaire :

- `0` : absence de réévaluation ;
- `1` : présence d’une réévaluation.

Le dépôt fournit :

- une préparation des données conçue pour limiter le data leakage sans identifiant patient ;
- une baseline TF-IDF + régression logistique ;
- un pipeline générique de fine-tuning Transformer, configuré actuellement avec DoctoBERT-fr ;
- un pipeline BioMistral-7B avec LoRA ;
- une validation croisée group-aware partagée entre les modèles ;
- une évaluation multi-seeds sur un jeu de test fixe ;
- des outils de comparaison, de bootstrap et de construction du rapport final ;
- une API FastAPI utilisant cinq modèles DoctoBERT de validation croisée.

Les données médicales et les poids des modèles ne sont pas versionnés. Le code de l’API est fourni, mais son bundle DoctoBERT doit être transmis séparément et installé localement avant de pouvoir lancer le service.

## 2. Démarrage rapide

Toutes les commandes doivent être exécutées depuis la racine du dépôt.

### 2.1. Récupérer le projet

```powershell
git clone <URL_DU_DEPOT>
cd projet-dereev
```

Le clone contient le code, mais ni les données ni les poids des modèles.

### 2.2. Créer l’environnement Python sous Windows

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
$env:PYTHONHASHSEED = "0"
```

La variable `PYTHONHASHSEED` garantit la reproductibilité et évite une relance du processus principal sous Windows.

### 2.3. Créer l’environnement sous Linux

```bash
python3 -m venv venv
source venv/bin/activate
python3 -m pip install --upgrade pip
python3 -m pip install -r requirements.txt
export PYTHONHASHSEED=0
```

Dans la suite du document, `python` peut être remplacé par `python3` selon l’environnement.

### 2.4. Vérifier l’installation

```powershell
python -c "import torch, transformers, sklearn, peft, fastapi; print('Dépendances OK')"
python -m src.tfidf_logistic --help
python -m main --help
python -m src.evaluate --help
python -m biomistral.main_biomistral --help
python -m biomistral.src.evaluate --help
```

### 2.5. État de l’API après le clone

L’API ne peut pas démarrer immédiatement après un clone, car ses poids ne sont pas inclus dans Git. L’erreur indiquant que le bundle `models/production/doctobert_fr_seed42` est introuvable est donc normale.

Pour activer l’API, récupérer le bundle auprès de l’équipe par un canal séparé et le placer exactement dans :

```text
models/production/doctobert_fr_seed42/
```

Une fois le bundle installé localement :

```powershell
python api.py
```

Puis ouvrir :

- interface simple : `http://127.0.0.1:8000/form` ;
- documentation Swagger : `http://127.0.0.1:8000/docs` ;
- état du service : `http://127.0.0.1:8000/health`.

## 3. Architecture du dépôt

```text
projet-dereev/
|-- README.md
|-- requirements.txt
|-- .gitignore
|-- main.py
|-- api.py
|-- build_benchmark.py
|-- equilibre.py
|-- src/
|   |-- config.py
|   |-- prepare_data.py
|   |-- preprocessing.py
|   |-- cv_protocol.py
|   |-- data_loader.py
|   |-- model.py
|   |-- evaluate.py
|   |-- tfidf_logistic.py
|   |-- threshold_metrics.py
|   |-- benchmark_statistics.py
|   |-- build_benchmark_report.py
|   |-- audit_hyperparameters.py
|   |-- compare_models.py
|   `-- plot_mean_pr_curves.py
|-- biomistral/
|   |-- main_biomistral.py
|   `-- src/
|       |-- config.py
|       |-- data_loader.py
|       |-- model.py
|       `-- evaluate.py
|-- ressources/
|   `-- abreviations/
|       `-- dico_global.json
|-- notebooks/
|   `-- eda.ipynb
|-- data/                         # local, non versionné
|   |-- raw/
|   `-- processed/
`-- models/
    |-- benchmark/                # sorties locales, non versionnées
    `-- production/
        `-- doctobert_fr_seed42/  # bundle API local, non versionné
```

Les dossiers `venv/`, `data/`, `models/benchmark/`, les caches Python et les résultats temporaires ne doivent pas être ajoutés à Git.

Le fichier local `arborescence.txt`, lorsqu’il existe, est un inventaire généré et n’est pas nécessaire au fonctionnement du projet.

## 4. Rôle de chaque fichier

### Fichiers racine

| Fichier | Rôle |
|---|---|
| `README.md` | Documentation de prise en main du projet. |
| `requirements.txt` | Dépendances Python de l’API, des modèles, des notebooks et du reporting. |
| `.gitignore` | Exclut les données, environnements, caches et modèles non destinés au dépôt. |
| `main.py` | Entraîne le Transformer défini dans `src/config.py` sur les folds fixes et plusieurs seeds. |
| `api.py` | API FastAPI de production utilisant l’ensemble DoctoBERT seed 42 composé de cinq folds. |
| `build_benchmark.py` | Agrège tous les modèles évalués, produit les tableaux, le classeur Excel et les graphes du benchmark. |
| `equilibre.py` | Ancien outil local de rééquilibrage d’un fichier brut. Il ne fait pas partie du protocole officiel actuel. |

### Package `src`

| Fichier | Rôle |
|---|---|
| `src/config.py` | Chemins, colonnes, seeds, modèle Transformer et hyperparamètres communs. |
| `src/prepare_data.py` | Détecte doublons/quasi-doublons, crée `leakage_group_id`, organise la revue humaine puis crée `train_cv.csv` et `test.csv`. |
| `src/preprocessing.py` | Nettoyage léger des textes et remplacement des abréviations. |
| `src/cv_protocol.py` | Crée ou recharge le manifeste de folds avec `StratifiedGroupKFold`. |
| `src/data_loader.py` | Charge et tokenise `cn_notes` et `reev_classe`. La colonne de groupe n’est jamais transmise au modèle. |
| `src/model.py` | Charge le Transformer Hugging Face avec une tête de classification binaire. |
| `src/evaluate.py` | Évalue les modèles Transformer sauvegardés sur le test fixe. |
| `src/tfidf_logistic.py` | Entraîne et évalue la baseline TF-IDF + régression logistique. C’est l’unique script TF-IDF à utiliser. |
| `src/threshold_metrics.py` | Calcule les métriques, les courbes précision-rappel et le seuil optimisé sur les OOF. |
| `src/benchmark_statistics.py` | Fonctions statistiques partagées, agrégations et intervalles bootstrap. |
| `src/build_benchmark_report.py` | Produit un rapport tabulaire CSV/XLSX à partir des résultats déjà évalués. |
| `src/audit_hyperparameters.py` | Vérifie et compare les hyperparamètres enregistrés par architecture. |
| `src/compare_models.py` | Comparaison appariée de deux modèles par Wilcoxon et bootstrap. |
| `src/plot_mean_pr_curves.py` | Compare graphiquement les courbes précision-rappel moyennes de deux modèles. |

### Package `biomistral`

| Fichier | Rôle |
|---|---|
| `biomistral/main_biomistral.py` | Entraînement BioMistral-7B-LoRA sur les folds group-aware partagés. |
| `biomistral/src/config.py` | Configuration spécifique BioMistral et LoRA. |
| `biomistral/src/data_loader.py` | Tokenisation spécifique à Mistral, avec padding à droite. |
| `biomistral/src/model.py` | Chargement de BioMistral, ajout des adaptateurs LoRA et contrôle CUDA/BF16. |
| `biomistral/src/evaluate.py` | Évaluation des adaptateurs BioMistral sur le test fixe. |

Les fonctions de folds, de métriques et de statistiques ne sont pas dupliquées dans `biomistral` : BioMistral réutilise les modules communs du package `src`.

### Ressources et notebooks

| Fichier | Rôle |
|---|---|
| `ressources/abreviations/dico_global.json` | Dictionnaire d’expansion des abréviations utilisé par le prétraitement. |
| `notebooks/eda.ipynb` | Analyse exploratoire. Il ne doit pas être utilisé pour modifier le test final après le benchmark. |

## 5. Données attendues

Le chemin brut par défaut est défini dans `src/config.py` :

```text
data/raw/dereev_data_4_deduplicated.xlsx
```

Ce fichier n’est pas fourni dans Git.

Colonnes attendues :

| Colonne | Obligatoire | Description |
|---|---:|---|
| `cn_notes` | oui | Texte de la note médicale. |
| `reev_classe` | oui | Label binaire `0` ou `1`. |
| `src` | non | Source documentaire. Si elle manque, la valeur `unknown` est utilisée. |

Les fichiers finaux contiennent :

```text
cn_notes,reev_classe,leakage_group_id
```

Ne jamais ajouter les fichiers `data/raw/`, `data/processed/`, les rapports contenant des notes ou les exports patients à Git.

## 6. Pourquoi `leakage_group_id` est indispensable

Le projet ne dispose pas d’un identifiant patient fiable. Une séparation aléatoire ligne par ligne pourrait donc placer dans deux ensembles différents :

- la même note dupliquée ;
- une version légèrement modifiée d’une note ;
- un copier-coller longitudinal ou copy-forward ;
- deux notes fortement apparentées provenant potentiellement du même contexte clinique.

Le modèle pourrait alors voir pendant l’entraînement un texte presque identique à un texte de validation ou de test. Les métriques seraient artificiellement optimistes : c’est du data leakage.

Le pipeline construit donc un identifiant de groupe documentaire nommé exactement :

```text
leakage_group_id
```

On peut le considérer comme l’« id_leak » du projet, mais ce n’est pas un identifiant patient. Il signifie seulement : « ces documents sont suffisamment proches pour rester ensemble lors des splits ».

Propriétés importantes :

- le label `reev_classe` n’est jamais utilisé pour décider si deux documents sont similaires ;
- les doublons exacts normalisés sont supprimés ;
- les quasi-doublons sont détectés à partir de similarités de caractères, de mots, de shingles et de tokens rares ;
- les dates, heures et nombres peuvent être masqués pour détecter le copy-forward ;
- la source, l’auteur et la proximité des dates ne sont que des signaux complémentaires ;
- les rapprochements incertains nécessitent une décision humaine ;
- toutes les notes d’un même groupe restent dans le même côté du split train/test et dans le même fold ;
- `leakage_group_id` sert uniquement aux splits et n’est jamais une feature du modèle.

L’identifiant a la forme :

```text
grp_<empreinte_sha256>
```

Il est construit à partir des identifiants stables des documents appartenant au groupe. Un groupe peut contenir des labels différents : cela ne justifie pas sa séparation, car l’objectif prioritaire est d’éviter la fuite documentaire.

Cette stratégie réduit fortement le risque de fuite sans prétendre reconstruire les patients. Sans véritable identifiant patient, une indépendance parfaite entre patients ne peut pas être garantie.

## 7. Préparation des données en deux étapes

La préparation doit rester en deux étapes. Il ne faut pas lancer directement une finalisation non revue sauf décision méthodologique explicite.

### 7.1. Étape 1 — Générer l’audit de similarité

```powershell
python -m src.prepare_data --stage review
```

Cette commande :

1. charge le fichier Excel ;
2. valide les colonnes et les labels ;
3. retire les notes vides ;
4. supprime les doublons exacts après normalisation ;
5. applique le nettoyage et le dictionnaire d’abréviations ;
6. calcule les candidats de similarité ;
7. fusionne automatiquement uniquement les paires à haute confiance ;
8. écrit les paires ambiguës à examiner manuellement.

Sorties principales :

```text
data/processed/leakage_audit/
|-- auto_merge_pairs.csv
|-- pairs_to_review.csv
|-- preliminary_group_summary.csv
|-- preliminary_grouped_dataset.csv
`-- grouping_configuration.json
```

Créer une copie du fichier de revue avant de le modifier. Cette copie conserve la trace des décisions même si l’audit est régénéré pendant `finalize`.

Windows :

```powershell
Copy-Item `
  data\processed\leakage_audit\pairs_to_review.csv `
  data\processed\leakage_audit\pairs_reviewed.csv
```

Linux :

```bash
cp \
  data/processed/leakage_audit/pairs_to_review.csv \
  data/processed/leakage_audit/pairs_reviewed.csv
```

Dans `pairs_reviewed.csv`, remplir la colonne `decision` :

- `merge` : les deux notes doivent appartenir au même groupe ;
- `separate` : les deux notes doivent rester dans des groupes différents.

La colonne `suggested_decision` est informative. Elle n’est jamais appliquée automatiquement comme décision humaine.

Ne pas modifier `left_document_id` ni `right_document_id`.

### 7.2. Étape 2 — Finaliser le split externe

```powershell
python -m src.prepare_data `
  --stage finalize `
  --review-file data\processed\leakage_audit\pairs_reviewed.csv `
  --overwrite
```

Équivalent Linux :

```bash
python3 -m src.prepare_data \
  --stage finalize \
  --review-file data/processed/leakage_audit/pairs_reviewed.csv \
  --overwrite
```

La finalisation :

- applique les fusions automatiques et les décisions `merge` ;
- construit les composantes documentaires et leurs `leakage_group_id` ;
- recherche un split externe group-aware proche de 15 % de test ;
- cherche à préserver les proportions de classes et de sources ;
- vérifie qu’aucun groupe ne traverse la frontière train/test ;
- produit un audit des similarités résiduelles entre train et test ;
- sauvegarde les anciennes sorties avant remplacement ;
- invalide les anciens manifests CV, devenus incompatibles avec le nouveau dataset.

Sorties officielles :

```text
data/processed/train_cv.csv
data/processed/test.csv
data/processed/leakage_audit/split_summary.json
data/processed/leakage_audit/train_cv_with_groups.csv
data/processed/leakage_audit/test_with_groups.csv
data/processed/leakage_audit/final_grouped_dataset.csv
data/processed/leakage_audit/residual_train_test_pairs.csv
```

Après une nouvelle finalisation, tous les modèles doivent être réentraînés.

### 7.3. Paramètres de `src.prepare_data`

| Paramètre | Défaut | Description |
|---|---|---|
| `--input` | valeur `RAW_XLSX` | Fichier Excel source. |
| `--output-dir` | `data/processed` | Répertoire de sortie. |
| `--stage` | `review` | `review` génère l’audit ; `finalize` crée les splits officiels. |
| `--review-file` | aucun | CSV contenant les décisions manuelles pour `finalize`. |
| `--allow-unreviewed` | désactivé | Autorise une finalisation avec des décisions vides. Déconseillé pour le protocole officiel. |
| `--overwrite` | désactivé | Sauvegarde puis remplace les sorties officielles et retire les manifests CV obsolètes. |
| `--text-column` | `cn_notes` | Nom de la colonne texte dans l’Excel. |
| `--label-column` | `reev_classe` | Nom de la colonne label. |
| `--source-column` | `src` | Nom de la colonne source. |
| `--test-size` | `0.15` | Fraction de test demandée, comprise entre `0.05` et `0.40`. |
| `--holdout-seed` | `42` | Seed de recherche du split externe. |

## 8. Split externe et folds partagés

La séparation s’effectue à deux niveaux différents :

```text
Dataset complet groupé
        |
        |-- split externe group-aware --> test.csv (jamais utilisé pour choisir le modèle)
        |
        `-- train_cv.csv
                |
                `-- StratifiedGroupKFold --> folds fixes de train/validation
```

`prepare_data.py` crée uniquement `train_cv.csv` et `test.csv`. Il ne crée pas les folds.

Lors du premier entraînement TF-IDF, Transformer ou BioMistral, `src/cv_protocol.py` crée :

```text
data/processed/cv_splits/fixed_split_seed_42_folds_5.csv
data/processed/cv_splits/fixed_split_seed_42_folds_5.json
```

Le manifeste est réutilisé par toutes les architectures et toutes les seeds. Il inclut une empreinte du texte, du label, du groupe et de l’ordre des lignes. Le programme refuse de réutiliser un manifeste incompatible après une modification des données.

Le protocole par défaut est :

```text
split externe : seed 42, environ 15 % de test
folds CV       : 5 folds group-aware, seed de split 42
entraînements  : seeds 42, 123, 456, 789 et 2026
total          : 5 folds x 5 seeds = 25 entraînements par architecture
```

Les seeds d’entraînement changent l’initialisation, le dropout et l’ordre des batchs, mais jamais la composition des folds.

## 9. Baseline TF-IDF + régression logistique

Le script officiel est uniquement :

```text
src/tfidf_logistic.py
```

### Pilote avec une seed

```powershell
python -m src.tfidf_logistic --folds 5 --seeds 42 --overwrite
```

### Benchmark complet

```powershell
python -m src.tfidf_logistic --folds 5 --overwrite
```

### Paramètres TF-IDF

| Paramètre | Défaut | Description |
|---|---:|---|
| `--folds` | `5` | Nombre de folds partagés. Minimum logique : 2. |
| `--seeds` | `42,123,456,789,2026` | Seeds séparées par des virgules. |
| `--ngram-max` | `2` | Taille maximale des n-grammes de mots. |
| `--max-features` | `50000` | Nombre maximal de features TF-IDF. |
| `--min-df` | `2` | Fréquence documentaire minimale. |
| `--max-df` | `0.98` | Fréquence documentaire maximale. |
| `--c` | `1.0` | Inverse de la régularisation de la régression logistique. |
| `--max-iter` | `2000` | Nombre maximal d’itérations du solveur. |
| `--bootstrap-resamples` | `2000` | Nombre de réplications bootstrap du test. |
| `--bootstrap-seed` | `2026` | Seed du bootstrap. |
| `--deployment-strategy` | `fold_ensemble` | `fold_ensemble` ou `single_model`. |
| `--overwrite` | désactivé | Remplace les résultats TF-IDF existants. |

Configuration interne actuelle :

```text
TfidfVectorizer : unigrammes + bigrammes, sublinear_tf=True
LogisticRegression : class_weight="balanced", solver="liblinear"
```

Sorties :

```text
models/benchmark/tfidf_lr/
|-- seed_<seed>/
|   |-- fold_1.joblib ... fold_5.joblib
|   |-- validation_predictions.csv
|   |-- test_predictions.csv
|   |-- decision_threshold.json
|   `-- experiment_summary.json
|-- all_seeds_test_results.csv
|-- all_fold_models_test_results.csv
|-- bootstrap_test_confidence_intervals.json
|-- benchmark_summary.json
`-- test_benchmark_summary.json
```

## 10. Transformer principal — DoctoBERT-fr

La configuration courante dans `src/config.py` est :

```text
Nom affiché       : DoctoBERT-fr-base
Checkpoint HF     : doctolib-lab/doctobert-fr-base
Dossier benchmark : models/benchmark/doctobert_fr_base
MAX_LENGTH        : 512
BATCH_SIZE        : 16
EPOCHS            : 12 maximum
LEARNING_RATE     : 2e-5
Early stopping    : patience 2, seuil 0.001
```

### Pilote avec une seed

```powershell
$env:PYTHONHASHSEED = "0"
python -m main --mode cv --folds 5 --seeds 42 --overwrite
python -m src.evaluate --folds 5 --seeds 42
```

Le pilote entraîne cinq modèles indépendants, un par fold.

### Benchmark complet

```powershell
$env:PYTHONHASHSEED = "0"
python -m main --mode cv --folds 5 --overwrite
python -m src.evaluate --folds 5
```

Le benchmark complet effectue 25 entraînements.

### Paramètres de `main.py`

| Paramètre | Défaut | Description |
|---|---|---|
| `--mode` | `cv` | Seule valeur acceptée ; conservée pour compatibilité. |
| `--folds` | `5` | Nombre de folds du manifeste partagé. |
| `--seeds` | `42,123,456,789,2026` | Seeds d’entraînement séparées par des virgules. |
| `--overwrite` | désactivé | Supprime uniquement les anciens résultats du modèle courant. Les manifests partagés sont conservés. |

### Paramètres de `src.evaluate`

| Paramètre | Défaut | Description |
|---|---|---|
| `--seeds` | seeds de `src/config.py` | Seeds à retrouver et évaluer. |
| `--folds` | `5` | Doit correspondre au nombre de folds entraînés. |
| `--bootstrap-resamples` | `2000` | Réplications bootstrap. |
| `--bootstrap-seed` | `2026` | Seed du bootstrap. |
| `--deployment-strategy` | `fold_ensemble` | Ensemble des folds ou proxy `single_model`. |

Le nombre de folds et les seeds d’évaluation doivent correspondre exactement aux dossiers entraînés.

Sorties principales :

```text
models/benchmark/doctobert_fr_base/
|-- run_configuration.json
|-- seed_<seed>/
|   |-- fold_1/ ... fold_5/
|   |-- validation_predictions.csv
|   |-- decision_threshold.json
|   `-- experiment_summary.json
|-- all_seeds_validation_results.csv
|-- all_seeds_test_results.csv
|-- all_fold_models_test_results.csv
|-- bootstrap_test_confidence_intervals.json
|-- strong_fold_disagreements.csv
|-- fold_disagreement_audit.xlsx
|-- benchmark_summary.json
`-- test_benchmark_summary.json
```

## 11. BioMistral-7B-LoRA

BioMistral est séparé du Transformer principal. Il réutilise les mêmes données, les mêmes groupes, le même manifeste CV et les mêmes fonctions statistiques.

Configuration actuelle :

```text
Checkpoint HF             : BioMistral/BioMistral-7B
Méthode                    : LoRA, pas QLoRA
Longueur maximale          : 512
Batch entraînement         : 2
Batch évaluation           : 4
Accumulation de gradients  : 8
Batch effectif             : 16
Époques maximum            : 12
Learning rate              : 2e-4
LoRA r / alpha / dropout   : 16 / 32 / 0.05
Modules LoRA               : q_proj, v_proj
Précision                   : BF16
```

`bitsandbytes` n’est pas nécessaire, car le code n’utilise ni quantification 4 bits ni quantification 8 bits.

### Vérifier le matériel

```powershell
python -c "import torch; print('CUDA:', torch.cuda.is_available()); print('BF16:', torch.cuda.is_bf16_supported() if torch.cuda.is_available() else False)"
```

L’entraînement exige un GPU NVIDIA visible par PyTorch et compatible BF16. En cas d’incompatibilité, le programme s’arrête immédiatement avec un message explicite.

### Pilote recommandé

```powershell
python -m biomistral.main_biomistral --folds 5 --seeds 42 --overwrite
python -m biomistral.src.evaluate --folds 5 --seeds 42
```

### Benchmark complet

```powershell
python -m biomistral.main_biomistral `
  --folds 5 `
  --seeds 42,123,456,789,2026 `
  --confirm-multi-seed `
  --overwrite

python -m biomistral.src.evaluate `
  --folds 5 `
  --seeds 42,123,456,789,2026
```

### Paramètres d’entraînement BioMistral

| Paramètre | Défaut | Description |
|---|---|---|
| `--mode` | `cv` | Seule valeur acceptée. |
| `--folds` | `5` | Nombre de folds partagés. |
| `--seeds` | `42` | Le mode pilote utilise volontairement une seed par défaut. |
| `--confirm-multi-seed` | désactivé | Obligatoire dès que plusieurs seeds sont demandées. |
| `--overwrite` | désactivé | Remplace les anciens résultats BioMistral. |

Les paramètres de `biomistral.src.evaluate` sont les mêmes que ceux de `src.evaluate`.

Les sorties se trouvent dans :

```text
models/benchmark/biomistral_7b_lora/
```

Chaque fold contient uniquement l’adaptateur LoRA et la tête de classification ; le modèle BioMistral-7B de base n’est pas recopié dans chaque dossier.

## 12. Métriques, OOF et seuils

Pour chaque seed, les modèles de folds prédisent uniquement leur fold de validation. L’assemblage de ces prédictions forme les prédictions out-of-fold, ou OOF.

Les OOF servent à :

- mesurer la validation sans prédire une ligne avec un modèle qui l’a vue ;
- choisir le seuil maximisant le F1 de la classe positive ;
- conserver le jeu de test totalement extérieur au choix du seuil.

Le pipeline rapporte notamment :

- F1 de la classe positive ;
- précision ;
- recall ;
- accuracy ;
- PR-AUC ;
- log loss lors de l’évaluation ;
- matrice de confusion ;
- intervalles bootstrap à 95 % sur le test.

Le seuil standard est `0.5`. Les scripts de benchmark peuvent aussi calculer un seuil optimisé sur les OOF entre `0.05` et `0.95`.

Important : l’API de production conserve volontairement un seuil fixe de `0.5`, même si un fichier de benchmark contient un autre seuil optimal.

Les résultats `moyenne ± écart-type` décrivent la variabilité entre seeds à split fixe. Les intervalles bootstrap décrivent séparément l’incertitude d’échantillonnage du test.

## 13. Construire et comparer le benchmark

### Agrégation complète automatique

```powershell
python build_benchmark.py
```

Le script découvre les dossiers `models/benchmark/*/all_seeds_test_results.csv` et produit notamment :

```text
models/benchmark/benchmark_comparison.csv
models/benchmark/benchmark_comparison_detailed.csv
models/benchmark/benchmark_results_by_seed.csv
models/benchmark/benchmark_results_by_fold.csv
models/benchmark/benchmark_bootstrap_ci.csv
models/benchmark/benchmark_hyperparameters.csv
models/benchmark/benchmark_warnings.csv
models/benchmark/benchmark_graphs.csv
models/benchmark/benchmark_comparison.xlsx
models/benchmark/graphs/
```

Seuls les modèles disposant des cinq seeds attendues doivent être présentés comme « benchmark complet ». Un pilote reste utile techniquement mais ne doit pas être classé comme résultat final.

### Rapport tabulaire modulaire

Tous les modèles évalués :

```powershell
python -m src.build_benchmark_report
```

Sélection :

```powershell
python -m src.build_benchmark_report `
  --models tfidf_lr,doctobert_fr_base,biomistral_7b_lora `
  --output-dir models\benchmark\report
```

### Comparaison statistique de deux modèles

```powershell
python -m src.compare_models `
  --model-a tfidf_lr `
  --model-b doctobert_fr_base `
  --bootstrap-resamples 5000 `
  --bootstrap-seed 2026
```

La différence est définie comme `modèle B - modèle A`. Les deux modèles doivent utiliser les mêmes observations de test et les mêmes seeds.

### Courbes précision-rappel moyennes

```powershell
python -m src.plot_mean_pr_curves `
  --model-a tfidf_lr `
  --model-b doctobert_fr_base `
  --grid-points 501 `
  --output-dir models\benchmark\pr_comparison
```

## 14. API de production

Les poids de l’API ne sont pas présents dans le dépôt Git. Sans installation manuelle du bundle, `python api.py` s’arrête volontairement avec un message indiquant que le dossier de modèles est introuvable. Ce comportement est attendu et ne signifie pas que l’installation Python est défectueuse.

Le bundle doit être transmis séparément par l’équipe, puis copié localement avec la structure suivante.

L’API utilise exclusivement :

```text
models/production/doctobert_fr_seed42/
|-- tokenizer/
|-- fold_1/
|-- fold_2/
|-- fold_3/
|-- fold_4/
|-- fold_5/
|-- deployment_manifest.json
|-- experiment_summary.json
`-- decision_threshold.json
```

Au démarrage, les cinq modèles sont chargés. Pour une note :

1. le texte est nettoyé avec le même prétraitement que le projet ;
2. chaque fold produit une probabilité de classe `1` ;
3. les cinq probabilités sont moyennées ;
4. la prédiction vaut `1` si la moyenne est supérieure ou égale au seuil fixe `0.5`.

Le seuil `0.5` est intentionnel et n’est pas remplacé par le contenu de `decision_threshold.json`.

### Lancer le serveur après installation du bundle

```powershell
python api.py
```

ou :

```powershell
python -m uvicorn api:app --host 127.0.0.1 --port 8000
```

### Routes

| Route | Méthode | Description |
|---|---|---|
| `/` | GET | Informations générales. |
| `/health` | GET | État, bundle, nombre de folds, seuil et appareil. |
| `/form` | GET | Interface HTML locale. |
| `/docs` | GET | Documentation Swagger générée par FastAPI. |
| `/predict` | POST JSON | Prédiction à partir d’un objet JSON. |
| `/predict-text` | POST texte | Prédiction à partir d’un corps `text/plain`. |

Exemple PowerShell :

```powershell
$body = @{
  texte = "Note médicale de test anonymisée"
  afficher_details = $true
} | ConvertTo-Json

Invoke-RestMethod `
  -Uri http://127.0.0.1:8000/predict `
  -Method Post `
  -ContentType "application/json" `
  -Body $body
```

Exemple `curl` :

```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"texte":"Note médicale de test anonymisée","afficher_details":true}'
```

Le chargement simultané des cinq folds nécessite plus de mémoire qu’un modèle unique. Les notes médicales ne doivent pas être envoyées vers une instance publique non sécurisée.

## 15. Modèles non versionnés et fichiers à exclure

Le dossier `models/` est entièrement ignoré par Git. Cela concerne :

- les checkpoints produits pendant les benchmarks ;
- les adaptateurs BioMistral ;
- le bundle de production utilisé par l’API ;
- les tokenizers sauvegardés avec les modèles.

Le dépôt ne contient ni ne distribue les poids. Ils doivent être conservés sur un stockage séparé autorisé par l’équipe, puis copiés manuellement sur la machine qui doit exécuter l’API.

Avant un commit, vérifier :

```powershell
git status --short
git check-ignore -v models/production/doctobert_fr_seed42/fold_1/model.safetensors
```

La deuxième commande doit confirmer que le poids est ignoré par `.gitignore`.

Ne pas ajouter :

- `data/raw/` ;
- `data/processed/` ;
- l’intégralité de `models/` ;
- `venv/` ;
- des notes médicales placées dans un script, notebook, log ou rapport ;
- `arborescence.txt` si ce fichier est seulement un inventaire local.

Les poids entraînés sur des notes médicales ne doivent être transmis que par un canal validé par l’équipe et selon les autorisations applicables.

## 16. Contrôles avant un entraînement long

### Vérification syntaxique

```powershell
python -m py_compile `
  main.py `
  api.py `
  src\prepare_data.py `
  src\cv_protocol.py `
  src\tfidf_logistic.py `
  src\evaluate.py `
  biomistral\main_biomistral.py `
  biomistral\src\evaluate.py
```

### Vérification des données et des groupes

```powershell
python -c "import pandas as pd; tr=pd.read_csv('data/processed/train_cv.csv'); te=pd.read_csv('data/processed/test.csv'); g='leakage_group_id'; print(tr.shape, te.shape); print('overlap=', len(set(tr[g]) & set(te[g]))); print(tr['reev_classe'].value_counts().sort_index()); print(te['reev_classe'].value_counts().sort_index())"
```

La valeur `overlap` doit être `0`.

### Ordre conseillé

1. vérifier ou produire les données en deux étapes ;
2. lancer TF-IDF avec une seed ;
3. lancer le Transformer avec une seed ;
4. évaluer cette seed ;
5. lancer ensuite les cinq seeds ;
6. construire le benchmark final ;
7. lancer BioMistral séparément sur un GPU compatible si nécessaire.

## 17. Erreurs fréquentes

### `ModuleNotFoundError`

Vérifier que l’environnement est activé et réinstaller les dépendances :

```powershell
.\venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

### L’API indique que le bundle est introuvable

C’est le comportement attendu après un clone : les poids ne sont pas versionnés. Récupérer séparément `doctobert_fr_seed42`, puis le placer dans `models/production/` avant de relancer `python api.py`.

### Le manifeste de folds est incompatible

Le contenu ou l’ordre de `train_cv.csv`, ses labels ou ses groupes ont changé. Ne pas forcer la réutilisation de l’ancien manifeste. Relancer la finalisation des données si nécessaire, puis laisser le premier entraînement recréer `data/processed/cv_splits/`.

### Des sorties existent déjà

Ajouter `--overwrite` uniquement si leur remplacement est volontaire. Pour `prepare_data`, les anciennes sorties sont sauvegardées. Pour les modèles, les résultats du modèle courant sont remplacés.

### BioMistral indique qu’aucun GPU CUDA n’est disponible

Le PyTorch installé est probablement une version CPU ou aucun GPU NVIDIA n’est accessible. Installer une version CUDA de PyTorch adaptée à la machine, puis vérifier :

```powershell
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
```

### BioMistral indique que BF16 n’est pas supporté

Le GPU n’est pas compatible avec le mode configuré. Ne pas désactiver arbitrairement BF16 sans valider la nouvelle configuration d’entraînement.

### `pairs_to_review.csv` contient encore des décisions vides

Compléter toutes les lignes dans la copie `pairs_reviewed.csv`, puis utiliser cette copie avec `--review-file`. `--allow-unreviewed` existe, mais contourne la revue humaine et doit rester exceptionnel.

## 18. Reproductibilité et règles méthodologiques

Pour comparer deux modèles correctement, conserver :

- le même `train_cv.csv` ;
- le même `test.csv` ;
- les mêmes `leakage_group_id` ;
- le même manifeste de folds ;
- `CV_SPLIT_SEED = 42` ;
- le même nombre de folds ;
- les mêmes seeds d’entraînement ;
- les mêmes définitions de métriques ;
- une sélection du seuil uniquement sur les OOF ;
- aucune sélection de modèle ou d’hyperparamètre sur le test.

Une reproductibilité bit à bit peut encore dépendre du GPU, de CUDA, de PyTorch, de Transformers et de certaines opérations natives. Les versions logicielles et le matériel utilisés doivent être consignés avec les résultats.


