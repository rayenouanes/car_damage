from __future__ import annotations

import os
from pathlib import Path
from urllib.parse import urlparse


def env_bool(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.lower() in {"1", "true", "yes", "on"}


def parse_s3_uri(uri: str) -> tuple[str, str]:
    parsed = urlparse(uri)
    if parsed.scheme != "s3" or not parsed.netloc or not parsed.path.strip("/"):
        raise ValueError(f"URI S3 invalide: {uri}")
    return parsed.netloc, parsed.path.lstrip("/")


def download_if_configured(s3_uri: str, destination: str, label: str) -> None:
    if not s3_uri:
        print(f"{label}: aucun URI S3 configure, etape ignoree")
        return

    destination_path = Path(destination).expanduser().resolve()
    force = env_bool("FORCE_MODEL_DOWNLOAD", False)
    if destination_path.is_file() and not force:
        print(f"{label}: deja present ({destination_path})")
        return

    try:
        import boto3
    except ImportError as exc:
        raise RuntimeError("boto3 est requis pour telecharger les modeles depuis S3") from exc

    bucket, key = parse_s3_uri(s3_uri)
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"{label}: telechargement s3://{bucket}/{key} -> {destination_path}")
    client = boto3.client("s3", region_name=os.getenv("AWS_REGION") or None)
    client.download_file(bucket, key, str(destination_path))


def main() -> None:
    download_if_configured(
        os.getenv("YOLO_MODEL_S3_URI", ""),
        os.getenv("YOLO_MODEL_PATH", "/app/models/best.pt"),
        "YOLO",
    )

    if env_bool("SAM2_ENABLED", True) and os.getenv("SAM2_PROVIDER", "mock").lower() == "sam2":
        download_if_configured(
            os.getenv("SAM2_CHECKPOINT_S3_URI", ""),
            os.getenv("SAM2_CHECKPOINT", "/app/models/sam2.pt"),
            "SAM2",
        )


if __name__ == "__main__":
    main()
