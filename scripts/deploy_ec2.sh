#!/usr/bin/env bash
# Deploy script for an Ubuntu EC2 instance (run on the server)
# Usage: ssh user@server 'bash -s' < deploy_ec2.sh
set -euo pipefail

echo "== Update & install prerequisites =="
sudo apt-get update -y
sudo apt-get install -y apt-transport-https ca-certificates curl gnupg lsb-release git

echo "== Install Docker =="
if ! command -v docker >/dev/null 2>&1; then
  curl -fsSL https://get.docker.com -o get-docker.sh
  sudo sh get-docker.sh
  rm get-docker.sh
  sudo usermod -aG docker $USER || true
fi

echo "== Install Docker Compose =="
if ! command -v docker-compose >/dev/null 2>&1; then
  sudo curl -L "https://github.com/docker/compose/releases/download/1.29.2/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
  sudo chmod +x /usr/local/bin/docker-compose
fi

echo "== Install Git LFS =="
if ! command -v git-lfs >/dev/null 2>&1; then
  curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh | sudo bash
  sudo apt-get install -y git-lfs
  git lfs install
fi

echo "== Clone repo and fetch LFS objects =="
REPO_URL="https://github.com/rayenouanes/car_damage.git"
APP_DIR="/opt/car_damage"
sudo rm -rf "$APP_DIR"
sudo mkdir -p "$APP_DIR"
sudo chown "$USER":"$USER" "$APP_DIR"
cd /tmp
if [ -d car_damage ]; then rm -rf car_damage; fi
git clone "$REPO_URL"
cd car_damage
git lfs install
git lfs pull

echo "== Copy to $APP_DIR =="
sudo rsync -a --delete . "$APP_DIR/"
cd "$APP_DIR"

if [ -f .env.example ] && [ ! -f .env ]; then
  cp .env.example .env
  echo "Created .env from .env.example. Edit .env to set secrets and model S3 URIs."
  exit 0
fi

echo "== Build and run docker compose =="
# Use gpu compose if available
if [ -f docker-compose.aws.yml ]; then
  docker compose -f docker-compose.aws.yml up -d --build
else
  docker compose up -d --build
fi

echo "Deployment finished. Use 'docker compose ps' and 'docker compose logs -f api' to inspect."