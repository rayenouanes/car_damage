# Plateforme Active Learning - Defauts automobiles

MVP local pour detecter et corriger quatre familles de defauts de carrosserie:

- `rayure`
- `bosse`
- `impact`
- `defaut_peinture`

Le chemin de production appelle uniquement YOLO et retourne classe, bbox et score. Le VLM Teacher, SAM2, le RAG et le LLM critique sont reserves a la boucle d'apprentissage. Un mode optionnel `audit/doute` peut appeler VLM/LLM pour les predictions faibles ou incoherentes.

## Architecture

```text
backend/app/
  main.py, config.py, database.py
  models/       schemas Pydantic et schema SQLite
  services/     video, tracking, keyframes, YOLO, SAM2, VLM, LLM, RAG, corrections, export, training
  adapters/     Ultralytics, SAM2/repli mock, mocks VLM/LLM, Chroma avec fallback
  routes/       upload, inference, correction, RAG, export
  data/         uploads, frames, predictions, exports, rag, error_bank
frontend/
  streamlit_app.py
tests/
```

L'application historique `app.py` reste independante de ce MVP. Son ecran Scanner accepte
une image, une video ou un lot d'images. Le mode video utilise YOLO + ByteTrack, choisit la
meilleure frame de chaque piste et applique SAM2 Tiny sur la bbox retenue.

Pour le mode video de `app.py`, les chemins SAM2 sont configurables avec
`SAM2_PACKAGE_ROOT`, `SAM2_CHECKPOINT`, `SAM2_MODEL_CONFIG` et `SAM2_DEVICE`. Sur une carte
de 4 Go, `SAM2_DEVICE=cpu` est recommande afin de reserver CUDA a YOLO/ByteTrack. Les bbox
restent dans `Data7.off/labels`; les polygones et masques sont ecrits separement dans
`Data7.off/labels_segmentation` et `Data7.off/masks`.

## Installation

Python 3.11 ou plus recent:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

ChromaDB est facultatif. Sans lui, `vector_store_chroma.py` utilise automatiquement un stockage JSON lexical testable:

```powershell
python -m pip install chromadb
$env:USE_CHROMA = "true"
```

## Modele YOLO

Par defaut, le backend cherche `best.pt` a la racine du depot. Un autre chemin peut etre configure:

```powershell
$env:YOLO_MODEL_PATH = "C:\modeles\carrosserie_best.pt"
```

Si aucun poids n'est present, le service active automatiquement un mock YOLO deterministe. Cela permet de tester tout le workflow sans telecharger de modele.

Le poids historique fourni contient six classes. L'adapter conserve les classes carrosserie, mappe `crack` vers `impact`, et filtre verre, phare et pneu. Un vrai poids quatre classes reste necessaire avant production.

## SAM2

SAM2 est un assistant d'annotation uniquement. Il ne fait jamais partie de l'inference production.

Configuration par defaut:

```powershell
$env:SAM2_ENABLED = "true"
$env:SAM2_PROVIDER = "mock"
```

Pour un vrai SAM2, installer le package SAM2 dans l'environnement, fournir un checkpoint local, puis configurer:

```powershell
$env:SAM2_ENABLED = "true"
$env:SAM2_PROVIDER = "sam2"
$env:SAM2_CHECKPOINT = "C:\modeles\sam2.pt"
$env:SAM2_MODEL_CONFIG = "sam2_hiera_l.yaml"
```

Si le package ou le checkpoint manque, le service revient automatiquement a `sam2_mock`. Avec `SAM2_ENABLED=false`, le pipeline continue avec des bbox; l'export segmentation les transforme en polygones rectangulaires de secours.

## Lancement

Backend FastAPI:

```powershell
python -m uvicorn backend.app.main:app --host 127.0.0.1 --port 8000
```

Swagger: <http://127.0.0.1:8000/docs>

Frontend Streamlit, dans un second terminal:

```powershell
python -m streamlit run frontend/streamlit_app.py
```

Interface: <http://127.0.0.1:8501>

Le raccourci suivant lance les deux processus:

```powershell
./start_active_learning_mvp.ps1
```

## Securite backoffice et API

En local, l'application reste ouverte si aucune variable de securite n'est definie.
Pour un deploiement partage, definir:

```powershell
$env:AL_API_KEY = "une_cle_longue"
$env:BACKOFFICE_USERNAME = "admin"
$env:BACKOFFICE_PASSWORD = "un_mot_de_passe_long"
```

Le backend exige alors l'en-tete `X-API-Key` sur les routes privees. `/health`
reste public pour les checks de supervision. Le frontend Streamlit envoie
automatiquement `AL_API_KEY` au backend et affiche une page de connexion si
`BACKOFFICE_PASSWORD` est configure.

## Deploiement AWS

Le repo contient les fichiers de base pour AWS:

```text
.env.example
Dockerfile
docker-compose.aws.yml
deploy_aws.md
scripts/download_models.py
```

Les modeles ne sont pas versionnes dans Git. Le conteneur peut les telecharger
depuis S3 au demarrage via `YOLO_MODEL_S3_URI` et `SAM2_CHECKPOINT_S3_URI`.

Commande de lancement sur EC2:

```bash
cp .env.example .env
docker compose -f docker-compose.aws.yml up -d --build
```

Voir `deploy_aws.md` pour les etapes EC2, S3, IAM, securite reseau et sauvegardes.

## Upload et inference

1. Ouvrir `Upload video/images`.
2. Choisir une video ou plusieurs images.
3. Selectionner une frame toutes les X frames ou toutes les X secondes.
4. Cliquer sur `Uploader`, puis `Lancer YOLO + active learning`.
5. Les videos sont stockees dans `backend/app/data/uploads/{job_id}` et leurs frames dans `backend/app/data/frames/{job_id}`.

La page `Analyse video + tracking` propose le workflow video dedie:

1. upload d'une video;
2. extraction toutes les X frames ou toutes les X secondes;
3. detection YOLO sur les frames nettes et non dupliquees;
4. regroupement temporel par classe, IoU, distance de centre et taille;
5. choix de la meilleure frame par piste selon confiance, nettete, luminosite, reflets et marge de bbox;
6. SAM2, VLM et LLM uniquement sur la bbox de cette meilleure frame.

## Selection intelligente des frames

Le pipeline n'envoie pas les frames brutes aux modeles couteux:

1. calcul de nettete par variance du Laplacien et rejet des frames floues;
2. hash perceptuel et rejet des doublons visuels;
3. conservation de la diversite visuelle/angles et des reflets forts;
4. prepassage YOLO leger sur les candidates;
5. creation de pistes temporelles et selection de la meilleure frame de chaque defaut;
6. SAM2, VLM et LLM uniquement sur les bbox de ces keyframes finales.

Les seuils sont configurables avec `FRAME_BLUR_THRESHOLD`, `FRAME_DUPLICATE_HAMMING`, `FRAME_ANGLE_HAMMING`, `FRAME_REFLECTION_RATIO`, `TRACKING_MAX_GAP`, `TRACKING_IOU_THRESHOLD` et `TRACKING_CENTER_DISTANCE`.

## Trois modes

- `training`: selection keyframes, YOLO, crops, SAM2, frames voisines, RAG, VLM, LLM et revue humaine.
- `audit/doute`: YOLO sur l'image demandee; VLM/LLM seulement sous le seuil ou en cas d'incoherence/reflet. SAM2 n'est pas appele.
- `production`: YOLO uniquement via `/api/production/infer`. Aucun SAM2, VLM, LLM, RAG ou Error Bank.

## Corriger une prediction

La page `File de review humaine` affiche l'image, YOLO, le masque SAM2, le VLM et le LLM. Elle permet d'accepter, valider le masque, rejeter, changer la classe, marquer reflet/salete/ombre, envoyer dans l'Error Bank et ajouter une regle RAG.

La page `Correction annotation` permet de modifier numeriquement la bbox et le polygone du masque. Chaque decision est enregistree en SQLite et sous forme JSON dans `backend/app/data/error_bank`.

## Export YOLO

1. Terminer les corrections du job.
2. Ouvrir `Export dataset YOLO`.
3. Configurer les ratios train et validation; test est calcule automatiquement.
4. Generer et telecharger le ZIP.

Le ZIP contient:

```text
images/train  images/val  images/test
labels/train  labels/val  labels/test
data.yaml
manifest.json
```

Toutes les frames d'une meme video partagent un `source_group` et sont affectees au meme split. Le manifeste expose `group_leakage`, qui doit rester `false`.

Les labels sont au format YOLO segmentation par defaut: `class_id x1 y1 x2 y2 ...`, avec coordonnees normalisees. Priorite du masque: correction humaine, proposition SAM2, puis rectangle bbox de secours.

## Reentrainement YOLO

L'interface et `/api/exports/{export_id}/train-yolo` preparent un entrainement segmentation. Par securite, il est desactive par defaut. Pour autoriser son execution:

```powershell
$env:YOLO_TRAINING_ENABLED = "true"
```

Le poids local configure par `YOLO_MODEL_PATH` sert de point de depart et les runs sont stockes dans `backend/app/data/training_runs`.

## Brancher un vrai VLM ou LLM

Pour Qwen2.5-VL ou LLaVA, creer un adapter avec la meme methode `analyze(...)` que `VLMTeacherMock`, puis l'injecter dans `VLMTeacherService` dans `main.py`.

Pour Ollama ou Transformers, creer un adapter avec la methode `critique(...)` de `LLMCriticMock`, puis l'injecter dans `LLMCriticService`.

Les sorties doivent rester des objets `VLMTeacherAnalysis` et `LLMCriticDecision`. Ne jamais importer ces providers dans la route `/api/production/infer` ou dans l'adapter YOLO.

## Tests

```powershell
python -m unittest discover -s tests -p "test_*.py" -v
```

Les tests couvrent extraction video, filtrage/keyframes, SAM2 mock/desactive, RAG, Error Bank, audit, isolation production et export segmentation anti-fuite.
