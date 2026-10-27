# LBP + kNN classification

Classification d'images par descripteurs **LBP** (Local Binary Patterns) et **plus proche voisin**, avec les distances SSD et Bhattacharyya.

Le projet est testé sur deux jeux de données :

- **CNR-EXT** : patches 150×150 de places de parking, à classer en libre ou occupée ;
- **ModelNet40** : objets 3D répartis en 40 catégories. Les 6 vues rendues de chaque objet sont assemblées en une seule image (montage).

## Principe

1. Pour chaque image, on calcule un histogramme LBP (rayon 1, 8 voisins). La variante multi-échelle concatène les histogrammes pour les rayons 1, 2 et 3.
2. Les histogrammes du jeu d'entraînement et du jeu de test sont enregistrés dans des fichiers `.txt`.
3. Chaque image de test reçoit l'étiquette de l'image d'entraînement la plus proche, selon la distance SSD ou Bhattacharyya.

Trois modes de lecture des images sont disponibles :

| Mode | Description |
|---|---|
| `grayscale` | image convertie en niveaux de gris |
| `packed_rgb` | canaux R, G et B placés côte à côte dans une seule image |
| `multiscale` | LBP multi-échelle (rayons 1, 2, 3) |

Pour ModelNet40, le jeu d'entraînement peut être **augmenté par des rotations aléatoires de ±5°** (Albumentations). Le script compare alors les résultats avec et sans ces images supplémentaires.

## Installation

```bash
pip install -r requirements.txt
```

## Données

Les jeux de données ne sont pas dans le dépôt, car ils sont trop volumineux. Il faut les télécharger et les placer à la racine du projet.

**CNRPark+EXT** : http://cnrpark.it/

```
PATCHES/   # images CNR-EXT-Patches-150x150 (OVERCAST/, RAINY/, SUNNY/)
LABELS/    # train.txt, test.txt, ... (une ligne = "chemin étiquette")
dataset2/
  train/empty/  train/occupied/
  test/empty/   test/occupied/
```

**ModelNet40, vues ombrées** (MVCNN) : https://github.com/suhangpro/mvcnn

```
shaded_modelnet40v1_train/
shaded_modelnet40v1_test/
```

## Utilisation

**Pipeline de base sur CNR-EXT** (`PATCHES/` + `LABELS/`) :

```bash
python lbp_knn_pipeline.py --train-max 5000 --test-max 5000
```

**Comparaison des modes sur `dataset2`** :

```bash
python run_dataset2.py --mode compare --train-max 1000 --test-max 1000
python run_dataset2.py --mode multiscale --show-misclassified --max-show 20
```

**Classification 3D sur ModelNet40** :

```bash
python run_3d_classifier.py                       # tous les objets, 1 rotation par objet
python run_3d_classifier.py --train-max 20 --test-max 10 --augmentations 0
python run_3d_classifier.py --show-misclassified --max-show 10
```

Pour `--train-max` et `--test-max`, la valeur `0` signifie « toutes les images » (ou tous les objets par catégorie pour la 3D).

**Benchmark** (précision et temps selon la taille des jeux, résultats écrits en CSV et PDF) :

```bash
python benchmark_lbp.py
```

## Résultats du benchmark (CNR-EXT, niveaux de gris, distance L1)

Extrait de [`benchmark_results.csv`](benchmark_results.csv) :

| Train | Test | Précision | Temps total |
|---:|---:|---:|---:|
| 100 | 1000 | 82,1 % | 1,6 s |
| 500 | 1000 | 87,6 % | 2,4 s |
| 1000 | 1000 | 89,3 % | 3,3 s |
| 2000 | 1000 | 90,1 % | 9,1 s |
| 5000 | 5000 | 92,0 % | 74,5 s |

La précision augmente avec la taille du jeu d'entraînement, puis se stabilise autour de 92 % à partir de 5000 images.

## Fichiers

| Fichier | Rôle |
|---|---|
| `lbp_knn_pipeline.py` | calcul des LBP, construction des descripteurs, classification (module partagé) |
| `run_dataset2.py` | comparaison des modes grayscale / packed_rgb / multiscale, SSD et Bhattacharyya |
| `run_3d_classifier.py` | montages 6 vues de ModelNet40, LBP multi-échelle, augmentation par rotation |
| `benchmark_lbp.py` | mesure de la précision et du temps pour plusieurs tailles de jeux |
