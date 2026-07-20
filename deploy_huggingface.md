# Deploiement permanent gratuit avec Hugging Face Spaces

Objectif: obtenir un lien stable du type:

```text
https://rayenouanes-car-damage.hf.space
```

Important: ce lien ne depend pas de ton PC et ne change pas a chaque relance. Il faut un compte Hugging Face.

## Pourquoi Hugging Face Spaces

- Streamlit Community Cloud est simple, mais ton projet contient des poids YOLO lourds ignores par Git.
- Cloudflare Quick Tunnel donne un lien temporaire.
- Cloudflare Tunnel permanent demande un domaine configure dans Cloudflare.
- Hugging Face Spaces donne un lien stable et accepte mieux les projets ML avec fichiers de modele.

## Limites importantes

- En gratuit, l'app tournera surtout sur CPU: la detection sera plus lente que sur ton PC GPU.
- Ne mets pas les datasets complets ni les annotations privees si le Space est public.
- Pour partager a tout le monde, garde seulement le modele de production, par exemple `best_2.pt`.

## Etapes

1. Cree un compte sur Hugging Face: https://huggingface.co
2. Cree un nouveau Space:
   - Name: `car-damage`
   - SDK: `Docker`
   - Visibility: `Public` si tu veux donner le lien a tout le monde
3. Copie dans le repo du Space le contenu du dossier `hf_space/`.
4. Ajoute aussi ton fichier modele:
   - `best_2.pt`
5. Dans les variables/secrets du Space, mets:

```text
BACKOFFICE_USERNAME=admin
BACKOFFICE_PASSWORD=admin2026
```

6. Le lien final sera:

```text
https://rayenouanes-car-damage.hf.space
```

Si ton nom Hugging Face est different, l'URL aura ce format:

```text
https://<username>-car-damage.hf.space
```

## Fichiers prepares

Le dossier `hf_space/` contient une version de deploiement qui lance directement `app.py` avec Streamlit.

Il faut y ajouter le modele `best_2.pt` avant de deployer.
