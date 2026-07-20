#!/usr/bin/env bash
set -euo pipefail

# Usage: run on the deployment server from the desired deploy directory
# Script will pull latest main, fetch LFS objects, create .env from example if missing, then start Docker Compose

REPO_DIR="/opt/car_damage"
REPO_URL="https://github.com/rayenouanes/car_damage.git"

if [ ! -d "$REPO_DIR" ]; then
  echo "Cloning repo into $REPO_DIR"
  sudo mkdir -p "$REPO_DIR"
  sudo chown "$USER":"$USER" "$REPO_DIR"
  git clone "$REPO_URL" "$REPO_DIR"
fi

cd "$REPO_DIR"

echo "Pull latest"
git fetch origin
git reset --hard origin/main

echo "Ensuring Git LFS objects"
git lfs install || true
git lfs pull || true

if [ -f .env.example ] && [ ! -f .env ]; then
  cp .env.example .env
  echo "Copied .env.example -> .env. Edit .env to set secrets before continuing."
  exit 0
fi

# Choose compose file
if [ -f docker-compose.aws.yml ]; then
  COMPOSE_FILE="docker-compose.aws.yml"
else
  COMPOSE_FILE="docker-compose.yml"
fi

echo "Using compose file: $COMPOSE_FILE"

# Build & run
docker compose -f "$COMPOSE_FILE" up -d --build

echo "Deployment launched. Use 'docker compose ps' and 'docker compose logs -f api' to inspect."