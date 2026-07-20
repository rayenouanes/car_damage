#!/usr/bin/env bash
# =============================================================================
# Script a executer UNE FOIS, depuis la racine du depot car_damage sur TON PC
# (la ou se trouvent reellement Data7.off, models_history/, runs/, etc.)
#
# Usage :
#   1. Copie ce fichier a la racine de ton depot car_damage
#   2. Ouvre un terminal (Git Bash sous Windows) dans ce dossier
#   3. Lance : bash push_dataset_et_modeles.sh
# =============================================================================
set -euo pipefail

echo "== 1/6 : Verification de Git LFS =="
if ! command -v git-lfs >/dev/null 2>&1; then
  echo "Git LFS n'est pas installe."
  echo "-> Windows : winget install GitHub.GitLFS   (ou https://git-lfs.com)"
  echo "-> macOS   : brew install git-lfs"
  echo "-> Linux   : sudo apt install git-lfs"
  echo "Installe-le puis relance ce script."
  exit 1
fi
git lfs install

echo "== 2/6 : Ecriture de .gitattributes (suivi LFS) =="
cat > .gitattributes << 'EOF'
# Images du dataset annote
*.jpg filter=lfs diff=lfs merge=lfs -text
*.jpeg filter=lfs diff=lfs merge=lfs -text
*.png filter=lfs diff=lfs merge=lfs -text
*.bmp filter=lfs diff=lfs merge=lfs -text
*.tif filter=lfs diff=lfs merge=lfs -text
*.tiff filter=lfs diff=lfs merge=lfs -text

# Videos eventuelles dans le dataset
*.mp4 filter=lfs diff=lfs merge=lfs -text
*.mov filter=lfs diff=lfs merge=lfs -text
*.avi filter=lfs diff=lfs merge=lfs -text

# Poids de modeles et checkpoints (historique de reentrainement)
*.pt filter=lfs diff=lfs merge=lfs -text
*.pth filter=lfs diff=lfs merge=lfs -text
*.onnx filter=lfs diff=lfs merge=lfs -text
*.ckpt filter=lfs diff=lfs merge=lfs -text

# Archives de sauvegarde
*.zip filter=lfs diff=lfs merge=lfs -text
EOF

echo "== 3/6 : Mise a jour de .gitignore (on ne cache plus le dataset/modeles) =="
cat > .gitignore << 'EOF'
active_learning_data/

# Environnements locaux
.venv/
.venv-gpu/
venv/
env/
.env
.env.local
.env.production
!.env.example

# Poids et artefacts ML lourds (desormais versionnes via Git LFS, voir .gitattributes)
# *.pt / *.pth / *.onnx / *.ckpt : suivis par LFS, plus ignores
*.engine
*.weights

# Datasets, annotations et exports locaux (desormais versionnes via Git LFS)
# Data7.off/, dataset_annote/, models_history/, runs/, evaluation_runs/,
# annotations_sessions/, backups_original_annotations/ : plus ignores, suivis par LFS
dataset_final_harmonise/
brain_training_dataset/
b2bak_*/
b2cur_*/

# Bases, caches et logs runtime
*.db
*.sqlite
*.sqlite3
*.log
*.tmp
results_*.csv
*_remap_report*.json
training_status.json
training_status.json.tmp
.pytest_cache/
__pycache__/
*.py[cod]

# Secrets Streamlit locaux
.streamlit/credentials.toml

backend/app/data/**
!backend/app/data/uploads/
!backend/app/data/uploads/.gitkeep
!backend/app/data/frames/
!backend/app/data/frames/.gitkeep
!backend/app/data/keyframes/
!backend/app/data/keyframes/.gitkeep
!backend/app/data/crops/
!backend/app/data/crops/.gitkeep
!backend/app/data/masks/
!backend/app/data/masks/.gitkeep
!backend/app/data/predictions/
!backend/app/data/predictions/.gitkeep
!backend/app/data/exports/
!backend/app/data/exports/.gitkeep
!backend/app/data/rag/
!backend/app/data/rag/.gitkeep
!backend/app/data/error_bank/
!backend/app/data/error_bank/.gitkeep
!backend/app/data/training_runs/
!backend/app/data/training_runs/.gitkeep
EOF

echo "== 4/6 : Correction du mot de passe expose dans deploy_streamlit_cloud.md =="
if [ -f deploy_streamlit_cloud.md ]; then
  sed -i.bak 's/BACKOFFICE_PASSWORD = ".*"/BACKOFFICE_PASSWORD = "change_moi_dans_streamlit_cloud_secrets"/' deploy_streamlit_cloud.md
  rm -f deploy_streamlit_cloud.md.bak
  echo "-> IMPORTANT : va aussi changer ce mot de passe reel dans Streamlit Cloud > Settings > Secrets,"
  echo "   il a ete visible publiquement sur GitHub jusqu'a present."
fi

echo "== 5/6 : Ajout et commit de tous les fichiers (dataset + modeles inclus) =="
git add .gitattributes .gitignore
[ -f deploy_streamlit_cloud.md ] && git add deploy_streamlit_cloud.md

for d in Data7.off dataset_annote models_history runs evaluation_runs annotations_sessions backups_original_annotations; do
  if [ -d "$d" ]; then
    echo "   -> ajout de $d/"
    git add "$d"
  else
    echo "   -> $d/ introuvable ici, ignore (verifie que tu es bien a la racine du depot)"
  fi
done

git commit -m "Versionne le dataset annote et l'historique des modeles via Git LFS" || echo "(rien a committer, ou deja a jour)"

echo "== 6/6 : Push vers GitHub =="
git push origin main

echo ""
echo "Termine. Verifie sur https://github.com/rayenouanes/car_damage que Data7.off"
echo "et tes modeles apparaissent bien (les gros fichiers afficheront 'Stored with Git LFS')."

# Instructions d'utilisation (Windows / Git Bash)
# 1) Placer ce fichier a la racine du depot (là où Data7.off est present)
# 2) Ouvrir Git Bash
# 3) Executer : bash push_dataset_et_modeles.sh

# Ce que fait le script :
# - Verifie que Git LFS est installe
# - Configure .gitattributes pour suivre les images et poids via LFS
# - Met a jour .gitignore pour ne plus exclure Data7.off, models_history/, runs/, etc.
# - Corrige un mot de passe expose dans deploy_streamlit_cloud.md
# - Ajoute, commit et push les gros fichiers vers GitHub (via LFS)
