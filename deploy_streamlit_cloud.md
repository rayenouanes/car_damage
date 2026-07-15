# Deploiement gratuit avec Streamlit Community Cloud

Objectif: obtenir un lien permanent Streamlit du type:

```text
https://car-damage.streamlit.app
```

## Principe

Le code reste sur GitHub. Le modele `best_2.pt` reste dans un depot Hugging Face prive et l'application le telecharge au demarrage avec un secret `HF_TOKEN`.

## Secrets a configurer dans Streamlit Cloud

Dans l'application Streamlit Cloud, ouvrir **Settings > Secrets** et ajouter:

```toml
BACKOFFICE_USERNAME = "admin"
BACKOFFICE_PASSWORD = "zlkeGxUzRkXAOhGimUgJw1FJ"
HF_TOKEN = "hf_xxx"
HF_MODEL_REPO = "rayeneouanes/car-damage-models"
HF_MODEL_FILENAME = "best_2.pt"
```

Ne mets jamais ces secrets dans GitHub.

## Configuration Streamlit Cloud

- Repository: `rayenouanes/car_damage`
- Branch: `main`
- Main file path: `app.py`

## Apres deploiement

1. Ouvrir le lien `.streamlit.app`.
2. Se connecter avec:
   - utilisateur: `admin`
   - mot de passe: la valeur `BACKOFFICE_PASSWORD`
3. Tester une image.

## Note

Sur l'offre gratuite, l'application tourne sur CPU. Les detections seront plus lentes que sur ton PC GPU.
