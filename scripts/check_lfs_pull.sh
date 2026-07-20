#!/usr/bin/env bash
set -euo pipefail

echo "== Vérification Git LFS et pull des objets LFS =="
if ! command -v git-lfs >/dev/null 2>&1; then
  echo "git-lfs manquant. Installez git-lfs puis relancez." >&2
  exit 2
fi

echo "Pull LFS..."
git lfs install
if git lfs pull; then
  echo "Git LFS pull terminé."
else
  echo "git lfs pull a échoué." >&2
  exit 3
fi

# Compter quelques artefacts
IMG_COUNT=$(find Data7.off/images -type f 2>/dev/null | wc -l || echo 0)
PT_COUNT=$(find . -maxdepth 2 -type f -name "*.pt" 2>/dev/null | wc -l || echo 0)

echo "Data7.off/images: ${IMG_COUNT} fichiers"
echo "Fichiers .pt trouvés (niveau racine/runs): ${PT_COUNT}"

if [ "${IMG_COUNT}" -eq 0 ]; then
  echo "Aucune image trouvée dans Data7.off/images — vérifie que tu as bien poussé depuis ta machine locale." >&2
  exit 4
fi

echo "Check LFS OK."