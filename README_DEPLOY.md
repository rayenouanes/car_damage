Deployment guide — Docker / AWS (recommended)

Overview
--------
This project is best deployed using Docker Compose on a server (VM or cloud). Large assets (dataset & model weights) are tracked with Git LFS.

Prerequisites (server)
- Docker & Docker Compose
- Git & Git LFS
- If using GPU: NVIDIA drivers + nvidia-docker (nvidia-container-toolkit)

1) Push dataset and weights from your workstation
-----------------------------------------------
On your local machine (where `Data7.off` and `.pt` files actually reside):

```bash
# from repo root on your PC
bash push_dataset_et_modeles.sh
```

This script configures Git LFS and pushes `Data7.off`, `annotations_sessions/`, `models_history/` and model `.pt` files to GitHub (stored with LFS).

2) Prepare the server
---------------------
SSH into the server and run:

```bash
git clone https://github.com/rayenouanes/car_damage.git
cd car_damage
git lfs install
git lfs pull
```

3) Create `.env` from `.env.example`
------------------------------------
Copy and fill secrets in `.env` (example keys):

```
AL_API_KEY=replace_me
BACKOFFICE_PASSWORD=replace_me
HF_TOKEN=hf_...
HF_MODEL_REPO=rayeneouanes/car-damage-models
YOLO_MODEL_PATH=/app/models/best_2.pt
```

4) Build and run with Docker Compose
------------------------------------
Use the provided `docker-compose.aws.yml` (GPU support) or the default `docker-compose.yml`.

```bash
# GPU-enabled stack (recommended for model inference/training)
docker compose -f docker-compose.aws.yml up -d --build

# or for local/dev
docker compose up -d --build
```

5) Verify services
------------------
Check containers and logs:

```bash
docker compose ps
docker compose logs -f api
```

Test API health (replace host/ip accordingly):

```bash
curl -s http://127.0.0.1:8000/health | jq
```

Open Streamlit UI at `http://<server-ip>:8501`.

6) Notes & troubleshooting
--------------------------
- If `ultralytics` or `cv2` fail during container startup, ensure the image has `opencv-python-headless` and a compatible `torch` installed. The Dockerfile should install packages from `requirements.txt`.
- For GPU, use images and `pip` wheels compatible with the server's CUDA version.
- If deploying on Streamlit Cloud instead of Docker, prefer hosting heavy weights on HuggingFace or S3 and set `HF_TOKEN` + `HF_MODEL_REPO`.

7) Updating model weights later
-------------------------------
- Re-train or place a new `best_2.pt` locally, then run `bash push_dataset_et_modeles.sh` from your PC to push the new weight (tracked with LFS).
- On the server do `git pull` and `git lfs pull` then restart the relevant container.

Contact
-------
If you want, I can:
- create an automated deployment script for AWS EC2 + Docker
- create a `.env.example` tailored to production (I checked `.env.example` exists)

