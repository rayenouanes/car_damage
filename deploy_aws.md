# Deploiement AWS - Car Damage Detection

Ce guide deploie le backoffice sur une instance EC2 GPU avec stockage persistant et modeles dans S3.

## Architecture conseillee

```text
GitHub repo rayenouanes/car_damage
  -> EC2 GPU
     -> Docker Compose
        -> api FastAPI : port 8000, prive sur localhost
        -> backoffice Streamlit : port 8501
     -> EBS / Docker volumes : donnees runtime, corrections, exports
  -> S3 : poids YOLO, checkpoint SAM2, exports sauvegardes
```

## 1. Preparer S3

Creer un bucket S3, par exemple:

```text
s3://your-car-damage-bucket
```

Uploader les poids:

```text
s3://your-car-damage-bucket/models/best_2.pt
s3://your-car-damage-bucket/models/sam2.1_hiera_tiny.pt
```

Ne pas mettre les poids dans GitHub.

## 2. Preparer EC2

Instance conseillee:

- demo / usage leger: `g4dn.xlarge`
- usage plus confortable: `g5.xlarge`

Regles reseau minimales:

- SSH `22`: uniquement votre IP ou VPN
- Streamlit `8501`: uniquement votre IP, VPN ou reverse proxy interne
- FastAPI `8000`: ne pas ouvrir publiquement, il est publie en local dans `docker-compose.aws.yml`

Attacher un volume EBS suffisant pour les videos, exports et corrections.

## 3. Role IAM EC2

Associer un role IAM a l'instance avec acces S3 limite au bucket du projet:

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Action": ["s3:GetObject", "s3:PutObject", "s3:ListBucket"],
      "Resource": [
        "arn:aws:s3:::your-car-damage-bucket",
        "arn:aws:s3:::your-car-damage-bucket/*"
      ]
    }
  ]
}
```

## 4. Installer Docker sur EC2

Sur Ubuntu:

```bash
sudo apt-get update
sudo apt-get install -y docker.io docker-compose-plugin git
sudo usermod -aG docker $USER
```

Se deconnecter puis se reconnecter.

Pour utiliser le GPU dans Docker, installer le NVIDIA driver et NVIDIA Container Toolkit selon l'image AMI choisie. Si vous utilisez une AMI Deep Learning AWS, les drivers sont souvent deja preinstalles.

## 5. Cloner le projet

```bash
git clone https://github.com/rayenouanes/car_damage.git
cd car_damage
cp .env.example .env
```

Modifier `.env`:

```text
AL_API_KEY=une_cle_longue_et_secrete
BACKOFFICE_USERNAME=admin
BACKOFFICE_PASSWORD=un_mot_de_passe_long
PUBLIC_BASE_URL=https://votre-url-ou-ip

AWS_REGION=eu-west-3
AWS_S3_BUCKET=your-car-damage-bucket
YOLO_MODEL_S3_URI=s3://your-car-damage-bucket/models/best_2.pt
YOLO_MODEL_PATH=/app/models/best_2.pt

SAM2_PROVIDER=mock
```

Pour activer le vrai SAM2 plus tard:

```text
SAM2_PROVIDER=sam2
SAM2_CHECKPOINT_S3_URI=s3://your-car-damage-bucket/models/sam2.1_hiera_tiny.pt
SAM2_CHECKPOINT=/app/models/sam2.1_hiera_tiny.pt
SAM2_MODEL_CONFIG=configs/sam2.1/sam2.1_hiera_t.yaml
```

## 6. Lancer

```bash
docker compose -f docker-compose.aws.yml up -d --build
```

Verifier:

```bash
docker compose -f docker-compose.aws.yml ps
curl http://127.0.0.1:8000/health
```

Ouvrir le backoffice:

```text
http://EC2_PUBLIC_IP:8501
```

## 7. Integration avec l'application du collegue

Pour un usage humain, partager uniquement:

```text
URL backoffice + login/mot de passe
```

Pour une integration applicative, partager:

```text
API URL
AL_API_KEY
Documentation endpoint
```

Exemple API production:

```bash
curl -X POST http://127.0.0.1:8000/api/production/infer \
  -H "X-API-Key: une_cle_longue_et_secrete" \
  -F "file=@car.jpg"
```

Cet exemple se lance depuis l'instance EC2. Pour l'application du collegue, exposer l'API
via un reverse proxy HTTPS prive, un VPN, un tunnel interne ou un load balancer protege,
pas directement sur Internet.

## 8. Sauvegardes

A sauvegarder:

- S3: modeles, exports, archives importantes
- EBS / volume Docker `app_data`: SQLite, Error Bank, uploads, frames, masks

Avant une mise a jour:

```bash
docker compose -f docker-compose.aws.yml down
git pull
docker compose -f docker-compose.aws.yml up -d --build
```

## 9. Commandes utiles

Logs:

```bash
docker compose -f docker-compose.aws.yml logs -f api
docker compose -f docker-compose.aws.yml logs -f backoffice
```

Relancer:

```bash
docker compose -f docker-compose.aws.yml restart
```

Verifier les modeles telecharges:

```bash
docker compose -f docker-compose.aws.yml exec api ls -lh /app/models
```
