# AGENTS.md

## Structure du repo

- `backend/app/main.py`: composition FastAPI et injection des services.
- `backend/app/config.py`: chemins et variables d'environnement.
- `backend/app/database.py`: acces SQLite; le DDL vit dans `models/db_models.py`.
- `backend/app/models/`: contrats Pydantic et schema de persistance.
- `backend/app/services/`: logique metier testable, dont tracking temporel, selection keyframes et SAM2.
- `backend/app/adapters/`: integrations externes ou mocks.
- `backend/app/routes/`: couche HTTP mince; pas de logique metier complexe.
- `backend/app/data/`: donnees runtime locales, ignorees par Git sauf `.gitkeep`.
- `frontend/streamlit_app.py`: client Streamlit de l'API.
- `tests/`: tests `unittest` par service.
- `app.py`: application historique; ne pas la modifier pour le MVP modulaire.

## Commandes

```powershell
python -m pip install -r requirements.txt
python -m uvicorn backend.app.main:app --host 127.0.0.1 --port 8000
python -m streamlit run frontend/streamlit_app.py
python -m unittest discover -s tests -p "test_*.py" -v
```

## Conventions

- Python 3.11+, annotations de types et Pydantic v2.
- Noms de classes metier canoniques en francais: `rayure`, `bosse`, `impact`, `defaut_peinture`.
- Les routes valident les entrees et deleguent aux services.
- Les services recoivent leurs dependances au constructeur pour rester testables.
- SQLite est la source de verite; les JSON de `predictions/` et `error_bank/` sont des traces auditables.
- Toute sortie VLM/LLM doit respecter son schema Pydantic.
- Le chemin production reste YOLO-only. Aucun appel RAG, VLM ou LLM dans `/api/production/infer`.
- SAM2 est interdit dans le chemin production et facultatif dans le mode audit.
- Le mode training execute SAM2, VLM et LLM sur toutes les bbox des keyframes retenues.
- Le prepassage YOLO peut voir les frames candidates; SAM2/VLM/LLM ne voient jamais toutes les frames brutes.
- Pour une video, le tracker choisit une meilleure frame par piste avant SAM2/VLM/LLM.
- Une correction humaine cree toujours un enregistrement Error Bank.
- Un export segmentation ne contient que des images dont toutes les detections ont ete revues.
- Les elements d'un meme `source_group` ne traversent jamais les splits.
- Ne pas versionner poids, uploads, frames, bases SQLite ou exports.

## Definition of done

Une modification est terminee lorsque:

1. les contrats API et metier demandes sont respectes;
2. les migrations/schema SQLite restent compatibles avec une base neuve;
3. les fallbacks fonctionnent sans poids YOLO, SAM2 ou ChromaDB;
4. les tests de services passent;
5. l'API `/health` demarre et le frontend se rend sans erreur;
6. la route production ne depend pas du VLM/LLM;
7. l'export produit des polygones YOLO segmentation, les six dossiers et `group_leakage=false`;
8. README et AGENTS sont actualises si les commandes ou l'architecture changent.
