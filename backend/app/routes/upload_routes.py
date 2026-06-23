from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, File, Form, HTTPException, Request, UploadFile

from backend.app.services.video_service import VIDEO_EXTENSIONS, safe_filename


router = APIRouter(prefix="/api/jobs", tags=["jobs-upload"])


@router.post("/upload")
async def upload_media(
    request: Request,
    files: list[UploadFile] = File(...),
    every_n_frames: int = Form(30),
    every_seconds: float | None = Form(None),
) -> dict:
    if not files:
        raise HTTPException(400, "Au moins un fichier est requis")
    names = [safe_filename(item.filename or "upload.bin") for item in files]
    kinds = {
        "video" if Path(name).suffix.lower() in VIDEO_EXTENSIONS else "image" for name in names
    }
    source_type = next(iter(kinds)) if len(kinds) == 1 else "mixed"
    database = request.app.state.database
    job_id = database.create_job(", ".join(names), source_type)
    count = 0
    warnings: list[str] = []
    for index, upload in enumerate(files):
        try:
            content = await upload.read()
            if not content:
                raise ValueError("Fichier vide")
            count += request.app.state.video_service.ingest(
                job_id, upload.filename or "upload.bin", content, index,
                every_n_frames=every_n_frames, every_seconds=every_seconds,
            )
        except Exception as exc:
            warnings.append(f"{upload.filename}: {exc}")
    if count == 0:
        message = "; ".join(warnings) or "Aucun media valide"
        database.update_job(job_id, "failed", message)
        raise HTTPException(400, message)
    return {"job_id": job_id, "images_created": count, "warnings": warnings}


@router.get("")
def list_jobs(request: Request) -> list[dict]:
    return request.app.state.database.list_jobs()


@router.get("/{job_id}")
def get_job(job_id: str, request: Request) -> dict:
    job = request.app.state.database.get_job(job_id)
    if not job:
        raise HTTPException(404, "Job introuvable")
    return job

